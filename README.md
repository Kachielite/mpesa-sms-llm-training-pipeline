# MPESA SMS LLM Training Pipeline

A comprehensive end-to-end pipeline for processing MPESA SMS transaction data and fine-tuning Large Language Models (LLMs). This project transforms raw SMS messages into structured training datasets and provides tools for fine-tuning models to understand and extract transaction information.

## 🎯 Project Overview

This project provides a complete machine learning pipeline for MPESA SMS transaction analysis. From raw SMS data to a trained LLM capable of extracting structured transaction information, this toolkit handles the entire workflow including data preprocessing, anonymization, formatting, and model fine-tuning.

### Key Features

- **Data Collection**: Load SMS messages from XML backups
- **Privacy Protection**: Comprehensive anonymization of personal information
- **Intelligent Parsing**: Extract key transaction fields using regex and NLP techniques
- **Flexible Formatting**: Support for both basic and instruct/chat model training formats
- **Cloud Integration**: Direct upload to Hugging Face Hub for dataset sharing
- **LLM Fine-tuning**: Complete pipeline for training models on MPESA transaction data *(coming soon)*
- **Model Evaluation**: Tools for assessing model performance on transaction extraction tasks *(coming soon)*

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
   - Copy `.env.example` to `.env` (if provided)
   - Add your Hugging Face token and desired repository ID

### Usage

1. **Prepare your data**: Place your SMS XML backup in the `data/` directory as `mpesa-sms.xml`

2. **Run the notebook**: Open `notebook.ipynb` and execute the cells sequentially:
   - Data Collection
   - Data Anonymization
   - Data Parsing
   - Data Formatting
   - Upload to Hugging Face Hub

3. **Output**: The processed data will be saved in the `output/` directory as:
   - `mpesa_basic.jsonl` - For basic model fine-tuning
   - `mpesa_instruct.jsonl` - For instruct/chat model fine-tuning

## 🧠 LLM Training (Coming Soon)

The pipeline will include:

- **Model Selection**: Support for popular open-source models (Llama, Mistral, etc.)
- **Fine-tuning Methods**: PEFT, QLoRA, and full fine-tuning options
- **Training Configurations**: Optimized hyperparameters for transaction extraction tasks
- **Model Evaluation**: Comprehensive metrics for assessing extraction accuracy
- **Deployment Tools**: Scripts for deploying trained models

## 📊 Pipeline Stages

1. **Data Preparation** ✅
   - SMS collection and loading
   - Privacy-preserving anonymization
   - Transaction field extraction
   - Dataset formatting

2. **Model Training** 🚧 *(In Development)*
   - Base model selection
   - Fine-tuning configuration
   - Training execution
   - Model validation

3. **Evaluation & Deployment** 📋 *(Planned)*
   - Performance metrics
   - Model comparison
   - Production deployment
   - API endpoints

## 📁 Project Structure

```
├── notebook.ipynb          # Main processing notebook
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables (create from .env.example)
├── .gitignore            # Git ignore rules
├── data/                 # Input data directory
│   └── mpesa-sms.xml    # SMS backup file
└── output/              # Generated datasets
    ├── mpesa_basic.jsonl
    └── mpesa_instruct.jsonl
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
  "input": "Anonymized SMS text...",
  "output": "{\"transaction_id\": \"...\", \"amount\": \"...\", ...}"
}
```

### Instruct Format
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Extract transaction details from the following SMS:\n\"...\""
    },
    {
      "role": "assistant", 
      "content": "{\"transaction_id\": \"...\", \"amount\": \"...\", ...}"
    }
  ]
}
```

## 🌐 Hugging Face Integration

The project includes seamless integration with Hugging Face Hub for dataset sharing and collaboration. Your processed datasets can be automatically uploaded and shared with the research community.

## 🛠️ Development

### Adding New Transaction Types

To add support for new transaction types:

1. Update the `transaction_types` list in `parse_mpesa_sms()` function
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
