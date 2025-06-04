<<<<<<< HEAD
<p align="center">
  <br/>
    <img alt="lighteval EPFL MNLP library logo" src="./assets/mnlp_lighteval_logo.png" style="max-width: 50%; max-height: 50%;">
  <br/>
</p>


<p align="center">
    <i>A go-to toolkit for flexible LLM evaluation, adapted from and inspired by Lighteval, RewardBench, & Langchain.</i>
</p>

<!-- <div align="center">

[![Tests](https://github.com/huggingface/lighteval/actions/workflows/tests.yaml/badge.svg?branch=main)](https://github.com/huggingface/lighteval/actions/workflows/tests.yaml?query=branch%3Amain)
[![Quality](https://github.com/huggingface/lighteval/actions/workflows/quality.yaml/badge.svg?branch=main)](https://github.com/huggingface/lighteval/actions/workflows/quality.yaml?query=branch%3Amain)
[![Python versions](https://img.shields.io/pypi/pyversions/lighteval)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/huggingface/lighteval/blob/main/LICENSE)
[![Version](https://img.shields.io/pypi/v/lighteval)](https://pypi.org/project/lighteval/)

</div>

--- -->

<!-- **Documentation**: <a href="https://huggingface.co/docs/lighteval/index" target="_blank">Lighteval's Wiki</a> -->

---

### LightEval - EPFL Modern Natural Language Processing 🚀

This evaluation suite is adapted from Huggingface's all-in-one toolkit for evaluating LLMs **Lighteval**, across multiple backends and benchmark types. Dive deep into your model’s performance by saving and exploring detailed, sample-by-sample results to debug and see how your models stack-up.

<!-- Customization at your fingertips: letting you either browse all our existing [tasks](https://huggingface.co/docs/lighteval/available-tasks) and [metrics](https://huggingface.co/docs/lighteval/metric-list) or effortlessly create your own [custom task](https://huggingface.co/docs/lighteval/adding-a-custom-task) tailored to your needs. -->


<!-- ## 🔑 Key Features

- **Speed**: [Use vllm as backend for fast evals](https://huggingface.co/docs/lighteval/use-vllm-as-backend).
- **Completeness**: [Use the accelerate backend to launch any models hosted on Hugging Face](https://huggingface.co/docs/lighteval/quicktour#accelerate).
- **Seamless Storage**: [Save results in S3 or Hugging Face Datasets](https://huggingface.co/docs/lighteval/saving-and-reading-results).
- **Python API**: [Simple integration with the Python API](https://huggingface.co/docs/lighteval/using-the-python-api).
- **Custom Tasks**: [Easily add custom tasks](https://huggingface.co/docs/lighteval/adding-a-custom-task).
- **Versatility**: Tons of [metrics](https://huggingface.co/docs/lighteval/metric-list) and [tasks](https://huggingface.co/docs/lighteval/available-tasks) ready to go. -->


## ⚡️ Installation

First, clone this repo to your home directory:

```bash
git clone https://github.com/eric11eca/lighteval-epfl-mnlp.git
```

Next, install from source with the `quantization` extras:
```bash
cd lighteval-epfl-mnlp
pip install -e .[quantization]  # on h100, you can `pip install -e .[quantization,quantization_fbgemm]` to install fbgemm package
```

Update the transformers version to `4.51.3`:
```bash
pip install transformers==4.51.3
```

<!-- Lighteval allows for many extras when installing, see [here](https://huggingface.co/docs/lighteval/installation) for a complete list. -->

If you want to push results to the Hugging Face Hub or access gated models & private datasets, add your access token as an environment variable:

```shell
huggingface-cli login
```

## 📋 Model Configs

To make model loading and configuration simple, set up a model config for each of the four models you need to implement for M2 and M3. All model config files are already set up for you in the directory `model_configs`: `dpo_model.yaml`, `mcqa_model.yaml`, `quantized_model.yaml`, `rag_model.yaml`. Please modify these files to reflect your model setup for each evaluation.

Inside the `rag_model.yaml` file, in addition to specifying the huggingface repo-id for your LLM, you also have to specify the huggingface repo-id for your document dataset (`docs_name_or_path`) and embedding model (`embedding_model`). For other arguments for RAG, please read the `rag_params` section in the config file for more details.

## 📝 Custom Tasks

To create a custom task that augments the Lighteval tasks, first, create a Python file under the `community_tasks` directory. We have put two example task files there already: `mnlp_dpo_evals.py` and `mnlp_mcqa_evals.py`. You can directly use these two task files for validation evaluation. If you want to evaluate on your dataset, please follow the two example files carefully.

If you want to create your evaluation data, make sure that the dataset follows exactly the format defined by the prompt functions (`preference_pair` & `mmlu_harness`). If you want to have your dataset format, please make sure that you define a new prompt function that will convert a line from your dataset to a document to be used for evaluation. You can then replace the input to the `prompt_function` argument with your newly defined function.

IMPORTANT NOTE: The metrics for MCQA and DPO evaluations have been set. Please do not modify the metrics! MCQA will always use `[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace]` and DPO will always use `[Metrics.reward_model_acc]`.


## 🚀 Launching Evaluation

To launch the evaluation, first set up the environment variables for accessing and caching with Huggingface:

```shell
# Often default to /home/<user>/.cache/huggingface/hub/
export HF_HOME=<path-to-your-hf-home-cache-dir>
# You can find this token in your user profile on HuggingFace Hub
export HF_TOKEN=<your-hf-hub-token>
```

Please use the following four commands to launch the evaluation of your four models with the Accelerate backend:

```shell
# Evaluating MCQA Model
lighteval accelerate \
    --eval-mode "lighteval" \
    --save-details \
    --override-batch-size <BATCH_SIZE> \
    --custom-tasks "community_tasks/mnlp_mcqa_evals.py" \
    --output-dir "<path-to-your-output-dir>" \
    model_configs/mcqa_model.yaml \
    "community|mnlp_mcqa_evals|0|0"

# Evaluating Quantized Model
lighteval accelerate \
    --eval-mode "lighteval" \
    --save-details \
    --override-batch-size <BATCH_SIZE> \
    --custom-tasks "community_tasks/mnlp_mcqa_evals.py" \
    --output-dir "<path-to-your-output-dir>" \
    model_configs/quantized_model.yaml \
    "community|mnlp_mcqa_evals|0|0"

# Evaluating DPO Model
lighteval accelerate \
    --eval-mode "dpo" \
    --save-details \
    --override-batch-size <BATCH_SIZE> \
    --custom-tasks "community_tasks/mnlp_dpo_evals.py" \
    --output-dir "<path-to-your-output-dir>" \
    model_configs/dpo_model.yaml \
    "community|mnlp_dpo_evals|0|0"

# Evaluating RAG Model
lighteval accelerate \
    --eval-mode "rag" \
    --save-details \
    --override-batch-size <BATCH_SIZE> \
    --custom-tasks "community_tasks/mnlp_mcqa_evals.py" \
    --output-dir "<path-to-your-output-dir>" \
    model_configs/rag_model.yaml \
    "community|mnlp_mcqa_evals|0|0"
```

## 📸 Logging

The evaluation will log the results automatically at the `output_dir` directory you specified, under the `results` directory. The results are saved in a sub-path `<HF_username>/<model_name>`. For example, with the model `meta-llama/Llama-3.2-1B-Instruct`, the results will be saved under `output_dir/results/meta-llama/Llama-3.2-1B-Instruct/`. Because we also set `save-details` details to be true. The sample-wise predictions will also be saved under the `details` directory with the same sub-path format. For example, `output_dir/details/meta-llama/Llama-3.2-1B-Instruct/`.
=======
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
>>>>>>> 0cb77cfe6c995798b0cd79d6510b97f164aa46aa
