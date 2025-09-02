#!/usr/bin/env python3
"""
MPESA LLM Fine-Tuning Script
Runs the complete training pipeline from the notebook
"""

import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

def main():
    print("🚀 Starting MPESA LLM Fine-Tuning Pipeline")

    # Load environment variables
    load_dotenv()

    # Login to services
    print("📋 Setting up authentication...")
    hf_token = os.getenv("HF_TOKEN")
    wandb_api_key = os.getenv("WANDB_API_KEY")

    if hf_token:
        login(token=hf_token)
        print("✅ Logged in to Hugging Face Hub")
    else:
        raise ValueError("HF_TOKEN not set in .env file")

    if wandb_api_key:
        wandb.login(key=wandb_api_key)
        print("✅ Logged in to Weights & Biases")
    else:
        raise ValueError("WANDB_API_KEY not set in .env file")

    # Initialize WandB
    wandb_project = "mpesa-llm-finetuning"
    wandb.init(project=wandb_project)
    print(f"✅ WandB initialized: {wandb_project}")

    # Load and prepare data
    print("📚 Loading and preparing MPESA SMS data...")
    DATA_PATH = "output/mpesa_basic.jsonl"

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    if os.path.getsize(DATA_PATH) == 0:
        raise ValueError(f"Data file is empty: {DATA_PATH}")

    # Load data
    raw = load_dataset("json", data_files=DATA_PATH)
    full_ds = raw["train"]
    split = full_ds.train_test_split(test_size=0.2, seed=42)
    train_ds, val_ds = split["train"], split["test"]

    print(f"📊 Loaded {len(full_ds)} examples. Split: {len(train_ds)} train, {len(val_ds)} test")

    # Format data for SFT
    def format_example_basic(ex):
        return {
            "text": (
                "### Task: Extract transaction details from the SMS\n"
                f"### Input:\n{ex['input']}\n"
                "### Output:\n" + ex["output"]
            )
        }

    train_text = train_ds.map(format_example_basic)
    val_text = val_ds.map(format_example_basic)

    print("✅ Data formatted for training")

    # Model setup
    print("🤖 Setting up model and tokenizer...")
    MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Check device
    if torch.backends.mps.is_available():
        dtype = torch.float16
        print("🍎 Using MPS (Apple Silicon) with float16")
    else:
        dtype = torch.float32
        print("💻 Using CPU with float32")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
    )

    print(f"✅ Model loaded on device: {next(model.parameters()).device}")

    # Configure LoRA
    print("⚙️ Configuring LoRA...")
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Training configuration
    print("📝 Setting up training configuration...")

    # Detect device capabilities for proper fp16/bf16 settings
    use_fp16 = False
    use_bf16 = False

    if torch.backends.mps.is_available():
        # On MPS, we can't use fp16 with Accelerate, so we disable it
        print("🍎 MPS detected - disabling fp16 for Accelerate compatibility")
        use_fp16 = False
        use_bf16 = False
    elif torch.cuda.is_available():
        # On CUDA, we can use fp16
        use_fp16 = True
        use_bf16 = False

    train_args = SFTConfig(
        output_dir="./mpesa-llm-mps",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        max_seq_length=256,
        fp16=use_fp16,  # Dynamically set based on device
        bf16=use_bf16,  # Dynamically set based on device
        report_to="wandb",
        eval_strategy="steps",
        eval_steps=200,
        dataset_text_field="text",
        # MPS-specific configurations
        dataloader_pin_memory=False,
        remove_unused_columns=True,  # Enable column removal to fix collation issues
        group_by_length=False,
    )

    # Create trainer with simplified configuration
    print("🏋️ Creating SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        train_dataset=train_text,
        eval_dataset=val_text,
        # Use default collator for now - simpler and more reliable
    )

    print("✅ Trainer created successfully!")

    # Start training
    print("🚀 Starting training...")
    trainer.train()

    print("✅ Training completed!")

    # Save model
    print("💾 Saving model...")
    trainer.save_model("./adapters")
    print("✅ Model saved to ./adapters")

    # Quick inference test
    print("🧪 Running quick inference test...")
    from transformers import pipeline

    gen = pipeline(
        "text-generation",
        model=trainer.model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=dtype,
    )

    sample = (
        "### Task: Extract transaction details from the SMS\n"
        "### Input:\nQFC3D45G7 Confirmed. You have received Ksh500 from Customer_1 XXXXXXX on 12/08/24 at 11:32 AM. New M-PESA balance is Ksh1,250.\n"
        "### Output:\n"
    )

    out = gen(sample, max_new_tokens=150, do_sample=False)
    output_text = out[0]["generated_text"][len(sample):]

    print("\n🎯 Inference Test Result:")
    print("-" * 50)
    print(output_text)
    print("-" * 50)

    if not output_text.strip():
        print("⚠️ Warning: Output is empty. Check your model or increase max_new_tokens.")
    elif len(output_text) >= 140:
        print("⚠️ Warning: Output may be truncated. Try increasing max_new_tokens.")
    else:
        print("✅ Inference test completed successfully!")

    print("\n🎉 Training pipeline completed successfully!")
    print("📁 Model saved in: ./adapters")
    print("📊 Training logs available in WandB")

if __name__ == "__main__":
    main()
