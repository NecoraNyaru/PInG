# finetune_cr.py

import os
import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, logging, set_seed
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from utils_dataset import build_cr_dataset, apply_chat_template
from utils_finetune import get_cr_args

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train(args):
    """
    Main function to run the SFT process for the Code Refinement task.
    """
    accelerator = Accelerator()
    set_seed(args.seed)

    # 1. Load Model and Tokenizer
    accelerator.print(f"Loading base model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        attn_implementation="flash_attention_2",
        device_map={"": accelerator.process_index},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=not args.gradient_checkpointing,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load and Prepare Dataset
    with accelerator.main_process_first():
        dataset = build_cr_dataset(args.dataset_name, args.dataset_subset)
        # Apply chat template
        dataset = dataset.map(
            lambda x: apply_chat_template(x['prompt'], x['completion'], tokenizer)
        )

    accelerator.print(f"Final dataset length: {len(dataset)}")
    if accelerator.is_main_process:
        print(f"Dataset example:\n{dataset[0]['text']}")

    # 3. Configure Training
    # [cite_start]As per the paper, the learning rate for code refinement is 2e-5 [cite: 166]
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        bf16=args.bf16,
        logging_strategy="steps",
        logging_steps=1,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        optim="adamw_torch_fused",
        save_strategy="epoch",
        report_to="none"
    )

    # Use a data collator to train only on the completion part of the prompt
    response_template = "<|im_start|>assistant"
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # 4. Initialize and Run Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        data_collator=data_collator,
    )
    
    accelerator.print(f"--- Starting Fine-Tuning for: Code Refinement with {args.model_path} ---")
    trainer.train(resume_from_checkpoint=args.resume)
    accelerator.print("--- Fine-Tuning Complete ---")

    # 5. Save Final Model
    if accelerator.is_main_process:
        final_save_path = os.path.join(args.output_dir, "checkpoint-final")
        trainer.save_model(final_save_path)
        tokenizer.save_pretrained(final_save_path)
        accelerator.print(f"Final model and tokenizer saved to {final_save_path}")


if __name__ == "__main__":
    args = get_cr_args()
    logging.set_verbosity_error()
    train(args)