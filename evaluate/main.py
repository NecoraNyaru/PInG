import os
import json
import argparse

from OpenCodeEval.args import get_args, check_args
from OpenCodeEval.utils import refine_text, write_jsonl, stream_jsonl, calculate_pass_at_k

from OpenCodeEval.factory import BenchmarkFactory, BackendFactory

from tqdm.contrib.concurrent import thread_map

def main():

    parser = argparse.ArgumentParser()
    args = get_args(parser)
    args = check_args(args)

    save_path = args.save_path
    os.makedirs(save_path, exist_ok = True)

    task = BenchmarkFactory.get_task(args)

    # Add a special execution path for the simulated user study
    if args.task.startswith("Simulated"):
        print(f"--- Running Refinement and Evaluation for {args.gittask} ---")

        # 1. Get prompts for the refinement model from the simulated data
        refinement_prompts = task.get_refinement_prompts()
        write_jsonl(os.path.join(args.save_path, "refinement_prompts.jsonl"), refinement_prompts)
        print(f"Constructed {len(refinement_prompts)} prompts for code refinement.")

        # 2. Generate the refined code parts using the specified LLM backend
        decoder = BackendFactory.get_backend(args)
        if decoder.is_chat():
            decoder.set_stop(task.chat_stop)
        else:
            decoder.set_stop(task.base_stop + task.chat_stop)

        print(f"Generating refined code using model: {args.model_name}...")
        args.num_samples = 1  # We only need one generation per prompt
        refinement_generations = decoder.generate(
            refinement_prompts,
            args.response_prefix,
            args.response_suffix
        )
        write_jsonl(os.path.join(args.save_path, "refinement_generations.jsonl"), refinement_generations)

        # 3. Post-process generations to construct the full, refined solutions
        full_solutions = thread_map(
            task.postprocess_refinement_generation,
            refinement_generations,
            max_workers=args.num_workers,
            desc="Constructing Full Solutions"
        )
        write_jsonl(os.path.join(args.save_path, "solutions.jsonl"), full_solutions)

        # 4. Evaluate the fully constructed solutions against the test cases
        evaluations = thread_map(
            task.process_results,
            full_solutions,
            max_workers=args.num_workers,
            desc="Evaluating Refined Solutions"
        )
        write_jsonl(os.path.join(args.save_path, "evaluations.jsonl"), evaluations)

        # 5. Calculate and display the final pass@1 rate
        num_passed = sum(1 for e in evaluations if e['passed'])
        total = len(evaluations)
        pass_rate = num_passed / total if total > 0 else 0
        
        final_results = {
            "refined_pass@1": pass_rate,
            "total_tasks": total,
            "passed_tasks": num_passed
        }
        print("\n--- Final Pass@1 Results for Refined Code ---")
        print(f"Pass@1: {pass_rate:.4f} ({num_passed}/{total})")
        
        results_path = os.path.join(args.save_path, "results.json")
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=4)
        print(f"\nAggregated results saved to {results_path}")

    else:
        # --- This is the original execution path for other benchmarks ---
        print(f"--- Running Evaluation for {args.task} ---")
        # get prompts
        prompts = task.get_prompt()

        for prompt in prompts:
            prompt['prompt'] = refine_text(args.prompt_prefix + prompt['prompt'] + args.prompt_suffix)
        prompts = sorted(prompts, key = lambda x: x['task_id'])
        write_jsonl(os.path.join(save_path, "prompts.jsonl"), prompts)

        if args.split  ==  'plus':
            base_path = save_path.replace('plus', 'base')
            if os.path.exists(os.path.join(base_path, "generations.jsonl")):
                generations = list(stream_jsonl(os.path.join(base_path, "generations.jsonl")))
                write_jsonl(os.path.join(save_path, "generations.jsonl"), generations)

        # check if generations exits
        if not os.path.exists(os.path.join(save_path, "generations.jsonl")):
            
            prompts = list(stream_jsonl(os.path.join(save_path, "prompts.jsonl")))
            # get generations
            decoder = BackendFactory.get_backend(args)
            if decoder.is_chat():
                decoder.set_stop(task.chat_stop)
            else:
                decoder.set_stop(task.chat_stop + task.base_stop)

            generations = decoder.generate(
                prompts,
                args.response_prefix,
                args.response_suffix
            )

            generations = sorted([data for data in generations if data['completion']], key = lambda x: (x['task_id'], x['completion_id']))
            write_jsonl(os.path.join(save_path, "generations.jsonl"), generations)

        else:
            
            generated_ids = [data['task_id'] for data in stream_jsonl(os.path.join(save_path, "generations.jsonl"))]
            prompts = [data for data in stream_jsonl(os.path.join(save_path, "prompts.jsonl")) if data['task_id'] not in generated_ids]
            if len(prompts) > 0:

                # get generations
                decoder = BackendFactory.get_backend(args)
                if decoder.is_chat():
                    decoder.set_stop(task.chat_stop)
                else:
                    decoder.set_stop(task.chat_stop + task.base_stop)

                continue_generations = decoder.generate(
                    prompts,
                    args.response_prefix,
                    args.response_suffix
                )

                generations = sorted([data for data in continue_generations if data['completion']] + list(stream_jsonl(os.path.join(save_path, "generations.jsonl"))), key = lambda x: (x['task_id'], x['completion_id']))
                write_jsonl(os.path.join(save_path, "generations.jsonl"), generations)


    # post-process generations
    # if not os.path.exists(os.path.join(save_path, "solutions.jsonl")):
    if True:

        generations = list(stream_jsonl(os.path.join(save_path, "generations.jsonl")))
        solutions = thread_map(
            task.postprocess_generation,
            generations,
            max_workers = args.num_workers,
            desc = "Post-processing Generations"
        )
        solutions = sorted(solutions, key = lambda x: (x['task_id'], x['completion_id']))
        write_jsonl(os.path.join(save_path, "solutions.jsonl"), solutions)

    # evaluate solutions
    # if not os.path.exists(os.path.join(save_path, "evaluations.jsonl")):
    if True:
        solutions = list(stream_jsonl(os.path.join(save_path, "solutions.jsonl")))
        evaluations = thread_map(
            task.process_results,
            solutions,
            max_workers = args.num_workers,
            desc = "Evaluating Solutions"
        )
        evaluations = sorted(evaluations, key = lambda x: (x['task_id'], x['completion_id']))
        write_jsonl(os.path.join(save_path, "evaluations.jsonl"), evaluations)

    # calculate pass@k
    # if not os.path.exists(os.path.join(save_path, "results.jsonl")):
    if True:
        evaluations = list(stream_jsonl(os.path.join(save_path, "evaluations.jsonl")))
        results = calculate_pass_at_k(evaluations, args.num_samples, args.list_k)
        write_jsonl(os.path.join(save_path, "results.jsonl"), results)


if __name__ == "__main__":
    main()