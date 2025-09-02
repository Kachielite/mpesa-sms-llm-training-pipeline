# MPESA SMS LLM Training Pipeline

A comprehensive end-to-end pipeline for processing MPESA SMS transaction data and fine-tuning Large Language Models (LLMs). This project transforms raw SMS messages into structured training datasets and provides complete tools for fine-tuning models to understand and extract transaction information.

## 🎯 Project Overview

This project provides a complete machine learning pipeline for MPESA SMS transaction analysis. From raw SMS data to a trained LLM capable of extracting structured transaction information, this toolkit handles the entire workflow including data preprocessing, anonymization, formatting, and model fine-tuning.

### Key Features

- **Data Collection**: Load SMS messages from XML backups
- **Privacy Protection**: Comprehensive anonymization of personal information
- **Intelligent Parsing**: Extract key transaction fields using regex and NLP techniques
- **Flexible Formatting**: Support for both basic and instruct/chat model training formats
- **Cloud Integration**: Direct upload to Hugging Face Hub for dataset sharing
- **LLM Fine-tuning**: Complete pipeline for training models on MPESA transaction data ✅
- **Model Evaluation**: Tools for assessing model performance on transaction extraction tasks
- **Mac M1 Optimized**: Specialized configuration for Apple Silicon training

## 📊 Extracted Fields

The system extracts the following key information from each SMS:

- `transaction_id` - Unique transaction identifier
- `amount` - Transaction amount in KSH
- `transaction_type` - Type of transaction (sent, received, withdrawn, airtime, etc.)
- `counterparty` - Other party involved in the transaction
- `date_time` - Transaction timestamp
- `balance` - Account balance after transaction

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)
- For training: Mac M1 with 16GB RAM (recommended) or similar hardware

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/mpesa-sms-llm-training-pipeline.git
cd mpesa-sms-llm-training-pipeline
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Create a `.env` file in the project root with the following variables:
   ```
   REPO_ID=your-huggingface-username/your-dataset-name
   HF_TOKEN=your_huggingface_token
   WANDB_API_KEY=your_wandb_api_key
   ```

### Usage

#### Data Preparation

1. **Prepare your data**: Place your SMS XML backup in the `data/` directory as `mpesa-sms.xml`

2. **Run data preparation**: Open `notebooks/data-preparation.ipynb` and execute the cells sequentially:
   - Data Collection from XML
   - Data Anonymization 
   - Data Parsing and field extraction
   - Data Formatting for training
   - Upload to Hugging Face Hub

3. **Output**: The processed data will be saved in the `output/` directory as:
   - `mpesa_basic.jsonl` - For basic model fine-tuning
   - `mpesa_instruct.jsonl` - For instruct/chat model fine-tuning

#### Model Training

1. **Configure training**: Open `notebooks/training.ipynb` 

2. **Run training pipeline**: Execute cells for:
   - Hugging Face and W&B authentication
   - Dataset loading and preparation
   - Base model selection (optimized for Mac M1)
   - LoRA configuration for efficient training
   - Supervised fine-tuning with TRL
   - Model saving and deployment

## 🧠 LLM Training

The pipeline includes complete LLM training capabilities:

- **Model Selection**: Optimized for Mac M1 (16GB RAM) with models like:
  - `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (recommended for 16GB)
  - `microsoft/Phi-3-mini-4k-instruct` 
  - `Qwen2-1.5B-Instruct`
- **Fine-tuning Methods**: LoRA/PEFT for memory-efficient training
- **Training Framework**: TRL (Transformers Reinforcement Learning) with SFT
- **Hardware Optimization**: Apple Silicon MPS acceleration
- **Experiment Tracking**: Weights & Biases integration
- **Model Management**: Automatic Hugging Face Hub integration

## 📊 Pipeline Stages

1. **Data Preparation** ✅
   - SMS collection and loading from XML
   - Privacy-preserving anonymization
   - Transaction field extraction
   - Dataset formatting for training

2. **Model Training** ✅
   - Base model selection and loading
   - LoRA/PEFT configuration
   - Supervised Fine-tuning (SFT)
   - Model validation and saving

3. **Model Deployment** ✅
   - Hugging Face Hub integration
   - Model merging and optimization
   - Inference testing

## 📁 Project Structure

```
├── notebooks/
│   ├── data-preparation.ipynb    # Data processing pipeline
│   └── training.ipynb           # LLM fine-tuning pipeline
├── requirements.txt             # Python dependencies
├── .env                        # Environment variables
├── .gitignore                  # Git ignore rules
├── data/
│   └── mpesa-sms.xml          # SMS backup file (XML format)
└── output/                    # Generated datasets
    ├── mpesa_basic.jsonl      # Basic training format
    └── mpesa_instruct.jsonl   # Instruct/chat format
```

## 🔒 Privacy & Security

This project takes privacy seriously:

- **Phone Number Masking**: All phone numbers are replaced with `XXXXXXX`
- **Name Anonymization**: Personal names are replaced with `CUSTOMER_XXXXXXXX` using random UUIDs
- **Agent Anonymization**: Agent numbers are formatted as `Agent_XXXX`
- **No Personal Data Storage**: Original personal information is not retained

## 🤖 Supported Transaction Types

The system recognizes and processes various MPESA transaction types:

- Money transfers (sent/received)
- Cash withdrawals
- Airtime purchases
- Buy goods and services
- Paybill payments
- Bill payments
- Deposits
- Reversals
- Fuliza transactions
- And more...

## 📈 Training Data Formats

### Basic Format
```json
{
  "input": "TAG2DHJIM8 Confirmed. Ksh460.00 sent to CUSTOMER_A31A50F3...",
  "output": "{\"transaction_id\": \"TAG2DHJIM8\", \"amount\": \"460.00\", \"transaction_type\": \"sent\", \"counterparty\": \"CUSTOMER_A31A50F3\", \"date_time\": null, \"balance\": \"5612.59\"}"
}
```

### Instruct Format
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Extract transaction details from the following SMS:\n\"TAG2DHJIM8 Confirmed. Ksh460.00 sent to CUSTOMER_A31A50F3...\""
    },
    {
      "role": "assistant", 
      "content": "{\"transaction_id\": \"TAG2DHJIM8\", \"amount\": \"460.00\", \"transaction_type\": \"sent\", \"counterparty\": \"CUSTOMER_A31A50F3\", \"date_time\": null, \"balance\": \"5612.59\"}"
    }
  ]
}
```

## 🌐 Hugging Face Integration

The project includes seamless integration with Hugging Face Hub for:
- Dataset sharing and versioning
- Model hosting and deployment
- Community collaboration
- Automated uploads from both notebooks

## 🛠️ Technical Details

### Dependencies
- `torch` - PyTorch for model training
- `transformers` - Hugging Face transformers library
- `datasets` - Dataset loading and processing
- `peft` - Parameter-Efficient Fine-Tuning (LoRA)
- `trl` - Transformers Reinforcement Learning library
- `wandb` - Experiment tracking
- `huggingface_hub` - Model and dataset management
- `python-dotenv` - Environment variable management

### Hardware Requirements
- **Recommended**: Mac M1/M2 with 16GB+ RAM
- **Alternative**: NVIDIA GPU with 8GB+ VRAM
- **Minimum**: 16GB system RAM for CPU-only training

### Training Configuration
- **Method**: LoRA (Low-Rank Adaptation) for efficient fine-tuning
- **Framework**: Supervised Fine-Tuning (SFT) with TRL
- **Optimization**: Apple Silicon MPS acceleration
- **Memory**: Float16 precision for reduced memory usage

## 🛠️ Development

### Adding New Transaction Types

To add support for new transaction types:

1. Update the `transaction_types` list in `parse_mpesa_sms()` function in data-preparation.ipynb
2. Add corresponding counterparty extraction logic if needed
3. Test with sample SMS messages

### Improving Anonymization

The anonymization logic can be extended in the `anonymize_sms()` function to handle additional sensitive information patterns.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📞 Support

If you encounter any issues or have questions:

1. Check the existing issues on GitHub
2. Create a new issue with detailed information
3. Include sample data (anonymized) if relevant

## ⭐ Acknowledgments

- Thanks to the open-source community for the tools and libraries used
- Special thanks to Hugging Face for providing the hub infrastructure
- Inspired by the need for privacy-preserving financial data analysis

---

**Note**: This tool is for research and educational purposes. Always ensure compliance with local data protection regulations and MPESA's terms of service when handling transaction data.
