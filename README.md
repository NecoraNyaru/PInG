# Enhancing Code Generation via Bidirectional Comment-Level Mutual Grounding

<p align="left">
    <a href="https://arxiv.org/abs/2505.07768"><img src="https://img.shields.io/badge/arXiv-2505.07768-b31b1b.svg?style=for-the-badge">
    <a href="https://opensource.org/license/mit/"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge">
</p>

This repository contains the official artifacts for the paper: 

**Enhancing Code Generation via Bidirectional Comment-Level Mutual Grounding**

*Yifeng Di, Tianyi Zhang*

2025 IEEE/ACM 47th International Conference on Software Engineering (ICSE 2025)

Our work introduces **PInG (Programming with Interactive Grounding)**, an interactive pipeline that helps establish a shared understanding between developers and Large Language Models (LLMs). By using inline comments as a medium for feedback, PInG allows for iterative refinement of generated code, leading to more accurate results that better align with developer intent.

## Repository Structure

This repository is organized to reflect the different stages of the PInG pipeline:

- `finetune/`: Contains scripts for fine-tuning the **Comment Generation** and **Code Refinement** models as described in the paper.

- `evaluate/`: The core evaluation framework used for **Initial Code Generation**, **Code Refinement**, and **Evaluation**.

- `comment/`: Contains the script for **Comment Generation**, which parses code snippets and adds comments using the fine-tuned model.

- `Enhancing_Code_Generation_via_Bidirectional_Comment_Level_Mutual_Grounding.pdf`: The PDF version of the paper.

- `README.md`: This file.

## Reproducing the Results

This section provides a step-by-step guide to reproduce the experiments from our paper.

### Step 1: Fine-tuning the Models
As described in the paper, PInG relies on two fine-tuned models.

1. **Comment Generation Model (CodeBERT)**:

    This model is fine-tuned on CodeSearchNet to generate comments for code statements.

    ```bash
    python finetune/finetune_cg.py --config ./configs/cg_config.json
    ```

    (You will need to create a `cg_config.json` file specifying the model path, dataset, and hyperparameters as detailed in the paper.)

2. **Code Refinement Model (DeepSeek Coder)**:

    This model is fine-tuned on The Stack to perform code refinement based on edited comments.

    ```bash
    python finetune/finetune_cr.py --config ./configs/cr_config.json
    ```

    (You will need to create a `cr_config.json` file specifying the model path, dataset, and hyperparameters as detailed in the paper.)

### Step 2: Generating Initial Code Solutions
Use the `evaluate` framework to generate initial (and potentially buggy) code solutions for a benchmark like HumanEval.

```bash
python evaluate/main.py \
    --model_name "deepseek-ai/deepseek-coder-6.7b-instruct" \
    --task HumanEval \
    --save_path ./output/deepseek-coder-6.7b-instruct/humaneval_iteration_0 \
    --num_workers 8 \
    --trust_remote_code
```

This will produce an `evaluations.jsonl` file in the `save_path` directory, containing the generated code snippets.

### Step 3: Generating Comments for the Code
Use the `generate.py` script to parse the code from the previous step and generate comments using your fine-tuned comment model.

```bash
python comment/generate.py \
    --input_file ./output/deepseek-coder-6.7b-instruct/humaneval_iteration_0/evaluations.jsonl \
    --output_file ./output/deepseek-coder-6.7b-instruct/humaneval_iteration_1/feedback.jsonl \
    --model_path ./models/finetuned_comment_model \
    --benchmark HumanEval \
    --benchmark_data_file ./evaluate/benchmark/data/HumanEval/base.jsonl \
    --initial_model "deepseek-ai/deepseek-coder-6.7b-instruct" \
    --iteration 1
```

The output `feedback.jsonl` now contains the code segmented into statements, each with a generated comment, ready for review.

### Step 4: Simulating Human Review (Manual Step)
This is a manual step where you provide feedback by editing the file generated in Step 3. Open `feedback.jsonl` and, for each incorrect code snippet, add a JSON object to the `annotator_feedback` array.

**Before (No Feedback):**
```json
{
    "task_id": "1",
    "iteration": 1,
    "...": "...",
    "code_with_comments": [
        {"statement": "digits = str(N)", "comment": "Convert N to a string"}
    ],
    "annotator_feedback": []
}
```

**After (Feedback Added):**
```json
{
    "task_id": "1",
    "iteration": 1,
    "...": "...",
    "code_with_comments": [
        {"statement": "digits = str(N)", "comment": "Convert N to a string"}
    ],
    "annotator_feedback": [
        {
            "statement_number": 2,
            "original_comment": "Convert N to a string",
            "edited_comment": "Convert N to a binary, and remove its prefix '0b'",
            "annotator_id": "annotator_1"
        }
    ]
}
```

### Step 5: Performing Code Refinement and Evaluation
Finally, use the `evaluate` framework again with a "Simulated" task to perform the code refinement and measure the improvement. The framework will use your fine-tuned refinement model to generate new code based on your feedback.

```bash
python evaluate/main.py \
    --task SimulatedHumanEval \
    --simulated_data_path ./output/deepseek-coder-6.7b-instruct/humaneval_iteration_1/feedback.jsonl \
    --save_path ./output/deepseek-coder-6.7b-instruct/humaneval_iteration_1 \
    --model_name /path/to/your/finetuned_refinement_model \
    --num_workers 8 \
    --trust_remote_code
```

The script will output the final `pass@1` score in the console and save the detailed evaluation results in the `./output/deepseek-coder-6.7b-instruct/humaneval_iteration_1` directory. To perform another round of feedback, you can use the `evaluations.jsonl` from this run as the input to Step 3 again.

## Citation

If you found our paper or this artifact useful in your research, please consider citing:

```bibtex
@inproceedings{di2025enhancing,
 title={Enhancing Code Generation via Bidirectional Comment-Level Mutual Grounding},
 author = {Di, Yifeng and Zhang, Tianyi},
 booktitle = {2025 IEEE/ACM 47th International Conference on Software Engineering (ICSE)},
 year = {2025},
} 
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.