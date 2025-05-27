# MNLP ChatSplaining

A comprehensive framework for fine-tuning language models on STEM content using multiple-choice question answering (MCQA) and open-answer formats. This project implements supervised fine-tuning (SFT), LoRA adaptation, and RAG (Retrieval-Augmented Generation) capabilities for educational AI applications.

## 🚀 Features

- **Multi-format Training**: Support for both MCQA and open-answer question formats
- **Advanced Fine-tuning**: LoRA adaptation and full supervised fine-tuning
- **RAG Integration**: PDF-based retrieval system for enhanced question answering
- **Smart Batching**: Token-budget aware batching for efficient GPU utilization
- **Comprehensive Evaluation**: Built-in evaluation metrics and visualization tools
- **Data Processing Pipeline**: Automated processing of multiple STEM datasets

## 📁 Project Structure

```
mnlp_chatsplaining/
├── main_rag.py              # RAG system for PDF-based Q&A
├── main_lora.py             # LoRA fine-tuning pipeline
├── main_pre-training.py     # Full supervised fine-tuning
├── utils/                   # Core utilities
│   ├── model_utils.py       # Model loading and configuration
│   ├── dataset_utils.py     # Dataset processing and formatting
│   ├── eval_utils.py        # Evaluation metrics and functions
│   ├── batching.py          # Smart token-budget batching
│   ├── push_to_hf.py        # Hugging Face Hub utilities
│   └── train_utils.py       # Training visualization tools
└── data/                    # Dataset processing scripts
    ├── base_processor.py    # Base class for MCQA datasets
    ├── base_openans_processor.py  # Base class for open-answer datasets
    └── [various]_processor.py     # Specific dataset processors
```

## 🔧 Installation

```bash
# Clone the repository
git clone <repository-url>
cd mnlp_chatsplaining

# Install dependencies
pip install torch transformers datasets peft
pip install langchain langchain-community faiss-cpu
pip install PyPDF2 jinja2 python-dotenv wandb
pip install matplotlib pandas tqdm psutil
```

## 🎯 Main Scripts

### 1. Supervised Fine-tuning (`main_pre-training.py`)

Full parameter fine-tuning with smart batching and comprehensive evaluation.

```bash
python main_pre-training.py \
    --dataset "RikoteMaster/OpenQA_merged" \
    --output_name "Qwen3-0.6B-SFT-OpenQA" \
    --num_train_samples 10000
```

**Features:**
- Token-budget aware batching for memory efficiency
- Support for both MCQA and open-answer datasets
- Automatic evaluation with sample predictions
- Training loss visualization
- Hugging Face Hub integration

### 2. LoRA Fine-tuning (`main_lora.py`)

Parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation).

```bash
python main_lora.py
```

**Features:**
- 4-bit quantization for memory efficiency
- ChatML format preprocessing
- Wandb integration for experiment tracking
- Automatic model pushing to Hugging Face Hub

### 3. RAG System (`main_rag.py`)

PDF-based retrieval-augmented generation for educational content.

```bash
# Rebuild embeddings from scratch
python main_rag.py --rebuild

# Normal operation
python main_rag.py
```

**Features:**
- PDF text extraction and smart chunking (512 tokens max)
- FAISS vector store with GPU acceleration
- Parallel PDF processing for faster indexing
- Metadata preservation (book title, chapter, page)

## 🛠️ Utilities

### Model Utils (`utils/model_utils.py`)
- **`load_model()`**: Unified model loading with 4-bit quantization support
- Support for various model configurations and device mapping

### Dataset Utils (`utils/dataset_utils.py`)
- **ChatML Format**: Structured prompt formatting with Jinja2 templates
- **`process_mcq_dataset()`**: MCQA format processing with chain-of-thought
- **`process_open_answer_dataset()`**: Open-answer format processing
- **`SFTDataCollator`**: Custom data collator with prompt masking

### Evaluation Utils (`utils/eval_utils.py`)
- **`evaluate_model_on_samples()`**: MCQA accuracy evaluation
- **`evaluate_openqa()`**: Qualitative evaluation for open-answer questions
- Test data loading from multiple sources

### Smart Batching (`utils/batching.py`)
- **`SmartPaddingTokenBatchSampler`**: Token-budget aware batching
- Power-of-2 batch size constraints
- Handles variable sequence lengths efficiently
- Prevents OOM errors through intelligent sample grouping

### Training Utils (`utils/train_utils.py`)
- **`plot_training_loss()`**: Training loss visualization
- Automatic figure saving and display

### Hugging Face Utils (`utils/push_to_hf.py`)
- **`DatasetUploader`**: Automated dataset uploading
- Format validation for MCQA and OpenQA
- Multi-split dataset support

## 📊 Data Processing

The `data/` folder contains processors for various STEM datasets:

### Base Classes
- **`BaseDatasetProcessor`**: Foundation for MCQA dataset processing
- **`BaseOpenQAProcessor`**: Foundation for open-answer dataset processing

### Supported Datasets

| Dataset | Type | Domain | Size | Processor |
|---------|------|---------|------|-----------|
| SciQ | MCQA | Science | 12K | `sciq_processor.py` |
| AQUA-RAT | MCQA | Math Reasoning | 60K | `aquarat_mcqa_processor.py` |
| MedReason | MCQA/OpenQA | Medicine | 32K | `medreason_processor.py` |
| OpenCode | OpenQA | Programming | 1.4M | `opencode_processor.py` |
| OpenMath | OpenQA | Mathematics | 3.2M | `openmath_processor.py` |
| CAMEL Chemistry | OpenQA | Chemistry | 20K | `camel_chemistry_openqa_processor.py` |
| StackExchange Engineering | OpenQA | Engineering | 36K | `stackexchange_engineering_openqa_processor.py` |

### Data Formats

**MCQA Format:**
```json
{
    "question": "What is machine learning?",
    "choices": ["A. AI subset", "B. Data analysis", "C. Programming", "D. Statistics"],
    "answer_index": 0,
    "answer_text": "A. AI subset",
    "source": "dataset_name",
    "explanation": "Machine learning is a subset of artificial intelligence..."
}
```

**OpenQA Format:**
```json
{
    "question": "Explain gradient descent",
    "answer": "Gradient descent is an optimization algorithm...",
    "source": "dataset_name",
    "explanation": "The algorithm works by..."
}
```

## 🎮 Usage Examples

### Training a Model on MCQA Data

```python
# Use main_pre-training.py for full fine-tuning
python main_pre-training.py \
    --dataset "jonlecumberri/MNLP_M2_mcqa_dataset" \
    --output_name "Qwen3-0.6B-MCQA" \
    --num_train_samples 5000
```

### LoRA Training

```python
# Use main_lora.py for parameter-efficient training
python main_lora.py
# Configuration is in the script - modify MODEL_NAME, DATASET_NAME, etc.
```

### Setting up RAG System

```python
# Initialize RAG with PDF books
rag = PDFRAG(max_chunk_size=512, similarity_threshold=0.1)

# Add PDF books
pdf_files = ["book1.pdf", "book2.pdf"]
rag.add_multiple_pdfs_parallel(pdf_files)

# Ask questions
result = rag.generate_answer("What is neural network backpropagation?")
print(result["answer"])
```

### Processing Custom Datasets

```python
from data.base_processor import BaseDatasetProcessor

class CustomProcessor(BaseDatasetProcessor):
    def process_dataset(self):
        # Implement your processing logic
        return train_data, val_data, test_data

processor = CustomProcessor()
processor.process_and_save()
processor.push_to_hub("your-repo/dataset-name")
```

## 📈 Training Configuration

### Memory Optimization
- **4-bit Quantization**: Reduces memory usage by ~75%
- **Gradient Checkpointing**: Trades compute for memory
- **Smart Batching**: Prevents OOM through token-budget management
- **Gradient Accumulation**: Effective large batch training

### Model Configuration
- **Base Model**: Qwen3-0.6B (efficient for educational tasks)
- **LoRA Config**: rank=8, alpha=16, dropout=0.05
- **Training**: BF16 precision, warmup steps, gradient clipping

## 🔍 Evaluation

### MCQA Evaluation
- Automatic accuracy calculation
- Letter-based answer matching
- Confidence scoring through retrieval scores

### OpenQA Evaluation
- Qualitative response generation
- Source attribution
- Context relevance scoring

## 🚀 Advanced Features

### Parallel Processing
- Multi-GPU support for training
- Parallel PDF processing for RAG
- Efficient data loading with smart batching

### Experiment Tracking
- Wandb integration for training metrics
- Loss curve visualization
- Model performance monitoring

### Production Ready
- Hugging Face Hub integration
- Containerization support
- Scalable architecture

## 📝 Environment Setup

Create a `.env` file:
```bash
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_key
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your dataset processor or utility
4. Test with existing pipelines
5. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on Hugging Face Transformers and Datasets
- Uses FAISS for efficient vector search
- Implements LoRA from Microsoft's PEFT library
- Evaluation framework inspired by LightEval

## 📞 Support

For questions and support, please open an issue in the repository or contact the development team.
