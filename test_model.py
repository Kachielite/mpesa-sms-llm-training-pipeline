#!/usr/bin/env python3
"""
Test the trained M-PESA model with proper PEFT inference
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def test_model():
    print("🔍 Testing M-PESA trained model...")

    # Load base model and tokenizer
    MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    ADAPTER_PATH = "./adapters"

    print("📥 Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check device
    if torch.backends.mps.is_available():
        dtype = torch.float16
        device = "mps"
        print("🍎 Using MPS (Apple Silicon)")
    else:
        dtype = torch.float32
        device = "cpu"
        print("💻 Using CPU")

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
    )

    # Load PEFT model
    print("🔗 Loading LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    print("✅ Model loaded successfully!")

    # Test cases
    test_cases = [
        {
            "name": "Received Money",
            "sms": "QFC3D45G7 Confirmed. You have received Ksh500 from Customer_1 XXXXXXX on 12/08/24 at 11:32 AM. New M-PESA balance is Ksh1,250.",
            "expected": {
                "transaction_id": "QFC3D45G7",
                "amount": "500",
                "transaction_type": "received",
                "counterparty": "Customer_1",
                "date_time": "12/08/24 at 11:32 AM",
                "balance": "1250"
            }
        },
        {
            "name": "Sent Money",
            "sms": "TAG2DHJIM8 Confirmed. Ksh460.00 sent to CUSTOMER_6D8DC478 16/1/25 at 11:18 AM. New M-PESA balance is Ksh5,612.59.",
            "expected": {
                "transaction_id": "TAG2DHJIM8",
                "amount": "460.00",
                "transaction_type": "sent",
                "counterparty": "CUSTOMER_6D8DC478",
                "date_time": "16/1/25 at 11:18 AM",
                "balance": "5612.59"
            }
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"🧪 Test {i}: {test_case['name']}")
        print(f"{'='*60}")

        # Format input
        prompt = (
            "### Task: Extract transaction details from the SMS\n"
            f"### Input:\n{test_case['sms']}\n"
            "### Output:\n"
        )

        print(f"📝 Input SMS: {test_case['sms']}")

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        if device == "mps":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        # Decode
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        print(f"🤖 Model Output:\n{response}")
        print(f"\n📋 Expected Output:\n{test_case['expected']}")

        # Simple evaluation
        if "transaction_id" in response and test_case['expected']['transaction_id'] in response:
            print("✅ Transaction ID: Found")
        else:
            print("❌ Transaction ID: Not found or incorrect")

        if test_case['expected']['amount'] in response:
            print("✅ Amount: Found")
        else:
            print("❌ Amount: Not found or incorrect")

        if "date_time" in response and response.count("null") == 0:
            print("✅ Date/Time: Attempted extraction")
        else:
            print("❌ Date/Time: Missing or null (training data issue)")

if __name__ == "__main__":
    test_model()
