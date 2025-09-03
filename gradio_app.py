#!/usr/bin/env python3
"""
Gradio UI for M-PESA SMS Transaction Extractor
Provides a web interface to test the trained model
"""

import gradio as gr
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class MPESAExtractor:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.load_model()

    def load_model(self):
        """Load the trained M-PESA model and tokenizer"""
        print("🔄 Loading M-PESA model...")

        # Model configuration
        MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        ADAPTER_PATH = "./adapters"

        # Check device
        if torch.backends.mps.is_available():
            dtype = torch.float16
            self.device = "mps"
            print("🍎 Using MPS (Apple Silicon)")
        else:
            dtype = torch.float32
            self.device = "cpu"
            print("💻 Using CPU")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            device_map="auto",
        )

        # Load PEFT model
        try:
            self.model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
            self.model.eval()
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading adapters: {e}")
            print("🔄 Using base model instead...")
            self.model = base_model
            self.model.eval()

    def extract_transaction(self, sms_text):
        """Extract transaction details from SMS text"""
        if not sms_text or not sms_text.strip():
            return json.dumps({
                "error": "Please enter an SMS message"
            }, indent=2)

        try:
            # Format prompt
            prompt = (
                "### Task: Extract transaction details from the SMS\n"
                f"### Input:\n{sms_text.strip()}\n"
                "### Output:\n"
            )

            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if self.device == "mps":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )

            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            # Clean up the response and try to parse as JSON
            response = response.strip()

            # Extract just the JSON part if there's extra text
            if response.startswith('{'):
                # Find the end of the JSON object
                brace_count = 0
                json_end = 0
                for i, char in enumerate(response):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break

                if json_end > 0:
                    response = response[:json_end]

            # Try to parse and pretty-print the JSON
            try:
                parsed_json = json.loads(response)
                return json.dumps(parsed_json, indent=2)
            except json.JSONDecodeError:
                # If parsing fails, return the raw response with formatting
                return f"Raw Output:\n{response}"

        except Exception as e:
            return json.dumps({
                "error": f"Error processing SMS: {str(e)}"
            }, indent=2)

# Initialize the extractor
extractor = MPESAExtractor()

def process_sms(sms_input):
    """Process SMS input and return extracted JSON"""
    return extractor.extract_transaction(sms_input)

# Create Gradio interface
def create_ui():
    """Create the Gradio interface"""

    # Custom CSS for better styling
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .input-container, .output-container {
        border-radius: 10px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    }
    """

    # Example SMS messages
    examples = [
        ["QFC3D45G7 Confirmed. You have received Ksh500 from Customer_1 XXXXXXX on 12/08/24 at 11:32 AM. New M-PESA balance is Ksh1,250."],
        ["TAG2DHJIM8 Confirmed. Ksh460.00 sent to CUSTOMER_6D8DC478 16/1/25 at 11:18 AM. New M-PESA balance is Ksh5,612.59."],
        ["TAH1K18FRF confirmed.You bought Ksh200.00 of airtime on 17/1/25 at 7:54 PM.New M-PESA balance is Ksh5,412.59."],
        ["TAI6MHU3IQ Confirmed. Ksh600.00 paid to CUSTOMER_64CDD32A. on 18/1/25 at 1:40 PM.New M-PESA balance is Ksh3,644.59."]
    ]

    # Create the interface
    with gr.Blocks(
        title="M-PESA SMS Transaction Extractor",
        theme=gr.themes.Soft(),
        css=css
    ) as demo:
        # Header
        gr.Markdown("""
        # 🏦 M-PESA SMS Transaction Extractor
        
        This AI-powered tool extracts structured transaction details from M-PESA SMS messages.
        Simply paste your M-PESA SMS message below and get the extracted details in JSON format.
        
        ### Extracted Fields:
        - **transaction_id**: Unique transaction identifier
        - **amount**: Transaction amount in Ksh  
        - **transaction_type**: Type (sent, received, airtime, paybill, etc.)
        - **counterparty**: Other party involved in transaction
        - **date_time**: Date and time of transaction
        - **balance**: Account balance after transaction
        """)

        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### 📝 Input SMS Message")
                sms_input = gr.Textbox(
                    label="M-PESA SMS Message",
                    placeholder="Paste your M-PESA SMS message here...",
                    lines=4,
                    elem_classes=["input-container"]
                )

                # Submit button
                submit_btn = gr.Button(
                    "🔍 Extract Transaction Details",
                    variant="primary",
                    size="lg"
                )

                # Clear button
                clear_btn = gr.Button(
                    "🗑️ Clear",
                    variant="secondary"
                )

            with gr.Column(scale=1):
                # Output section
                gr.Markdown("### 📊 Extracted Transaction Details")
                json_output = gr.Code(
                    label="JSON Output",
                    language="json",
                    elem_classes=["output-container"],
                    lines=15
                )

        # Examples section
        gr.Markdown("### 📱 Try These Examples")
        gr.Examples(
            examples=examples,
            inputs=[sms_input],
            outputs=[json_output],
            fn=process_sms,
            cache_examples=False
        )

        # Footer
        gr.Markdown("""
        ---
        **Note**: This model was trained on anonymized M-PESA SMS data. 
        Personal information in SMS messages is automatically anonymized for privacy protection.
        """)

        # Event handlers
        submit_btn.click(
            fn=process_sms,
            inputs=[sms_input],
            outputs=[json_output]
        )

        clear_btn.click(
            fn=lambda: ("", ""),
            outputs=[sms_input, json_output]
        )

    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_ui()

    print("🚀 Starting M-PESA SMS Transaction Extractor UI...")
    print("🌐 The interface will be available at: http://localhost:7860")

    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True if you want a public Gradio link
        quiet=False
    )
