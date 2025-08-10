import os
from transformers import (
    RobertaTokenizer,
    EncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    logging,
)
from utils_dataset import get_cg_dataset
from utils_finetune import get_cg_args

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train(args):
    """
    Main function to run the Seq2Seq fine-tuning process for CodeBERT.
    """
    set_seed(args.seed)

    # 1. Load Tokenizer and Model
    print(f"Loading tokenizer and model from: {args.model_path}")
    tokenizer = RobertaTokenizer.from_pretrained(args.model_path)
    
    # For seq2seq tasks, we initialize CodeBERT as an encoder-decoder model.
    # This uses CodeBERT's weights for both the encoder and the decoder.
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        args.model_path, args.model_path, tie_encoder_decoder=True
    )

    # Set special tokens for the decoder
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # 2. Load and Prepare Dataset
    # The dataset utility handles loading, filtering, and tokenizing
    train_dataset = get_cg_dataset(
        dataset_name=args.dataset_name,
        subset_name=args.dataset_subset,
        tokenizer=tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length
    )
    print(f"Dataset prepared. Total samples: {len(train_dataset)}")

    # 3. Configure Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="epoch",
        predict_with_generate=True,
        seed=args.seed,
        report_to="none"
    )

    # 4. Initialize and Run Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
    )

    print(f"--- Starting Fine-Tuning for: Comment Generation with {args.model_path} ---")
    trainer.train(resume_from_checkpoint=args.resume)
    print("--- Fine-Tuning Complete ---")

    # 5. Save Final Model
    final_save_path = os.path.join(args.output_dir, "checkpoint-final")
    trainer.save_model(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"Final model and tokenizer saved to {final_save_path}")

if __name__ == "__main__":
    args = get_cg_args()
    logging.set_verbosity_error()
    train(args)