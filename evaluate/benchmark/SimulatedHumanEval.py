import os
from typing import Literal

from .HumanEval import HumanEval
from ..utils import stream_jsonl
from ..eval.func_eval import check_correctness
from ..eval.sanitize import sanitize

class SimulatedHumanEval(HumanEval):
    """
    A specialized Benchmark class to run the PInG simulated user study.
    It orchestrates a refine-then-evaluate workflow:
    1. Constructs refinement prompts from user feedback data.
    2. Post-processes the generated code to form a complete solution.
    3. Evaluates the final solution against HumanEval test cases.
    """

    name: str = "SimulatedHumanEval"

    def __init__(
        self,
        split: Literal["base", "plus"] = "base",
        time_out: float = 3.0,
        prompt_type: str = "Completion",
        data_path: str = None
    ):
        super(HumanEval, self).__init__()
        self.split = split
        self.time_out = time_out
        self.prompt_type = prompt_type

        if not data_path:
            raise ValueError("The '--simulated_data_path' must be provided for the simulated task.")
        self.simulated_data_path = data_path

        # Load original HumanEval data for prompts and test cases
        self.original_humaneval_tasks = super().get_task()
        # Load the simulated interaction data
        self.simulated_tasks = self._get_simulated_tasks()

    def _get_simulated_tasks(self):
        """Loads the simulated user study data from the provided JSONL file."""
        tasks = {}
        for task_data in stream_jsonl(filename=self.simulated_data_path):
            task_id = int(task_data["task_id"].split("/")[-1])
            tasks[task_id] = task_data
        return tasks
    
    def get_refinement_prompts(self):
        """
        Constructs prompts for the refinement model based on the annotator feedback.
        The prompt includes the original task, the correct code before the edit,
        and the user's edited comment as the final instruction.
        """
        prompts = []
        for task_id, sim_data in self.simulated_tasks.items():
            feedback_list = sim_data.get('annotator_feedback')
            # Skip if there is no feedback
            if not feedback_list:
                continue
            
            # Find the feedback with the earliest statement number to start the refinement
            feedback = min(feedback_list, key=lambda f: f['statement_number'])

            statement_idx = feedback['statement_number'] - 1  # Convert to 0-based index
            edited_comment = feedback['edited_comment']
            original_task_prompt = self.original_humaneval_tasks[task_id]['prompt']

            # Reconstruct the correct part of the code (the "head") from the statement list
            code_head_statements = [
                sim_data['code_with_comments'][i]['statement'] 
                for i in range(statement_idx)
            ]
            code_head = '\n'.join(code_head_statements)

            # Get the original statement that is being replaced to determine its indentation.
            original_incorrect_statement = sim_data['code_with_comments'][statement_idx]['statement']

            # Calculate the leading whitespace from that specific statement.
            indentation = ' ' * (len(original_incorrect_statement) - len(original_incorrect_statement.lstrip(' ')))
            
            # As per the PInG paper, the new context is:
            # original context + code up to the refined comment + refined comment
            refinement_prompt = (
                f"{original_task_prompt.strip()}\n"
                f"{code_head}\n"
                f"{indentation}# {edited_comment}"
            )

            prompts.append({
                "task_id": task_id,
                "prompt": refinement_prompt,
                "completion_id": 0 # Only one sample per prompt
            })
        return prompts

    def postprocess_refinement_generation(self, generation: dict):
        """
        Combines the newly generated code tail with the original code head
        to create the full, refined solution for evaluation.
        """
        task_id = generation['task_id']
        sim_data = self.simulated_tasks[task_id]
        generated_code_tail = generation['completion']

        # Find the feedback with the earliest statement number to start the refinement
        feedback_list = sim_data.get('annotator_feedback')
        feedback = min(feedback_list, key=lambda f: f['statement_number'])

        statement_idx = feedback['statement_number'] - 1

        # Reconstruct the head of the code again
        code_head_statements = [
            sim_data['code_with_comments'][i]['statement'] 
            for i in range(statement_idx)
        ]
        code_head = '\n'.join(code_head_statements)

        # The full refined solution is the combination of the correct head
        # and the newly generated tail.
        full_solution = f"{code_head}\n{generated_code_tail}"
        
        # Sanitize the complete solution to extract valid, reachable code
        entry_point = self.original_humaneval_tasks[task_id]["entry_point"]
        sanitized_solution = sanitize(full_solution, entry_point)

        return {
            "task_id": task_id,
            "completion_id": "refined_0",
            "solution": sanitized_solution
        }
