import os
from typing import Literal

from .mbpp import mbpp
from ..utils import stream_jsonl, refine_text
from ..eval.func_eval import check_correctness
from ..eval.sanitize import sanitize

class SimulatedMbpp(mbpp):
    """
    A specialized Benchmark class to run the PInG simulated user study on the mbpp dataset.
    """
    name: str = "SimulatedMbpp"

    def __init__(
        self,
        split: Literal["full", "sanitized"] = "sanitized",
        time_out: float = 3.0,
        prompt_type: str = "Instruction",
        data_path: str = None
    ):
        super(mbpp, self).__init__(split=split, time_out=time_out, prompt_type=prompt_type)
        if not data_path:
            raise ValueError("The '--simulated_data_path' must be provided for the simulated task.")
        self.simulated_data_path = data_path
        
        # Load original mbpp tasks for prompts and tests
        self.original_mbpp_tasks = super().get_task()
        # Load simulated interaction data
        self.simulated_tasks = self._get_simulated_tasks()

    def _get_simulated_tasks(self):
        """Loads the simulated data from the provided JSONL file."""
        tasks = {}
        for task_data in stream_jsonl(filename=self.simulated_data_path):
            task_id = int(task_data["task_id"].split("/")[-1])
            tasks[task_id] = task_data
        return tasks

    def get_refinement_prompts(self):
        """Constructs prompts for the refinement model."""
        prompts = []
        for task_id, sim_data in self.simulated_tasks.items():
            feedback_list = sim_data.get('annotator_feedback')
            # Skip if there is no feedback
            if not feedback_list:
                continue

            # Find the feedback with the earliest statement number
            feedback = min(feedback_list, key=lambda f: f['statement_number'])

            statement_idx = feedback['statement_number'] - 1
            edited_comment = feedback['edited_comment']
            
            original_task_data = self.original_mbpp_tasks[task_id]
            # Use the original format_prompt from the parent mbpp class, but without the code part
            original_task_prompt = super().format_prompt(
                promblem=original_task_data["text"],
                tests=original_task_data["test_list"]
            ).replace("\n```python\n", "") # Remove the default code block starter
            
            # Construct the code head
            code_head_statements = [sim_data['code_with_comments'][i]['statement'] for i in range(statement_idx)]
            code_head = '\n'.join(code_head_statements)
            
            # Determine indentation from the original incorrect statement
            original_incorrect_statement = sim_data['code_with_comments'][statement_idx]['statement']
            indentation = ' ' * (len(original_incorrect_statement) - len(original_incorrect_statement.lstrip(' ')))
            
            # The refinement prompt includes the original problem, few-shot examples (if any), the code head, and the new comment
            refinement_prompt = (
                f"{self.few_shots_prompt}\n" if self.few_shots_prompt else ""
                f"{original_task_prompt.strip()}\n"
                f"```python\n"
                f"{code_head}\n"
                f"{indentation}# {edited_comment}"
            )

            prompts.append({
                "task_id": task_id,
                "prompt": refine_text(refinement_prompt),
                "completion_id": 0
            })
        return prompts

    def postprocess_refinement_generation(self, generation: dict):
        """Combines the generated code tail with the code head to form a full solution."""
        task_id = generation['task_id']
        sim_data = self.simulated_tasks[task_id]
        generated_code_tail = generation['completion']
        
        # Clean up the generated tail from markdown code blocks
        if "```" in generated_code_tail:
             generated_code_tail = generated_code_tail.split("```")[0].strip()

        # Find the feedback with the earliest statement number
        feedback_list = sim_data.get('annotator_feedback')
        feedback = min(feedback_list, key=lambda f: f['statement_number'])
        statement_idx = feedback['statement_number'] - 1

        code_head_statements = [sim_data['code_with_comments'][i]['statement'] for i in range(statement_idx)]
        code_head = '\n'.join(code_head_statements)

        full_solution = f"{code_head}\n{generated_code_tail}"
        
        # The mbpp sanitize function doesn't require an entry point
        sanitized_solution = sanitize(full_solution)

        return {
            "task_id": task_id,
            "completion_id": "refined_0",
            "solution": sanitized_solution
        }