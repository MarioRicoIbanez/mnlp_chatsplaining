#!/usr/bin/env python3
"""
PDF RAG system main script using LangChain components for processing and retrieving information from PDF books.
Features:
- LangChain PyMuPDFLoader/PyPDFLoader for fast PDF text extraction
- Smart chunking with max size of 256 tokens using SentenceTransformersTokenTextSplitter
- Structured prompt templates with ChatPromptTemplate/PromptTemplate
- Enhanced retrieval with LangChain FAISS vector store
- Simplified workflow with Document objects and metadata preservation
"""

import argparse
import time
from pathlib import Path
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR  # Assuming the script is in the project root

# Set up project directories relative to script location
RESULTS_DIR = PROJECT_ROOT / "results_model"
RAG_DIR = PROJECT_ROOT / "RAG"
EMBEDDINGS_DIR = RESULTS_DIR / "embeddings"

# Create necessary directories
RESULTS_DIR.mkdir(exist_ok=True)
RAG_DIR.mkdir(exist_ok=True)
EMBEDDINGS_DIR.mkdir(exist_ok=True)

# Load environment variables from .env file (look in script directory first)
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    load_dotenv()  # Try default locations

# Import RAG utilities
from utils.rag_utils import PDFRAG

# Import PDF utilities for document extraction
from utils.pdf_utils import (
    extract_documents_from_single_pdf,
    extract_documents_from_multiple_pdfs,
    check_pymupdf_installation,
)


def create_rag_instance(use_custom_embedding: bool = False, max_chunk_size: int = 256) -> PDFRAG:
    """Create a PDFRAG instance with LangChain integration and specified configuration.
    
    Args:
        use_custom_embedding: Whether to use custom trained embeddings
        max_chunk_size: Maximum tokens per chunk
        
    Returns:
        Configured PDFRAG instance with LangChain components
    """
    if use_custom_embedding:
        custom_model_path = EMBEDDINGS_DIR / "custom_model"
        if not custom_model_path.exists():
            logger.error(f"No custom model found at {custom_model_path}")
            logger.info("Train a custom model first with: python main_rag.py --mode train")
            raise FileNotFoundError(f"Custom model not found at {custom_model_path}")
        
        logger.info(f"Using custom embedding model from: {custom_model_path}")
        return PDFRAG(
            max_chunk_size=max_chunk_size,
            similarity_threshold=0.1,
            use_custom_embedding=True,
            custom_embedding_path=str(custom_model_path),
            project_root=PROJECT_ROOT,
            rag_dir=RAG_DIR,
            results_dir=RESULTS_DIR,
            embeddings_dir_path=EMBEDDINGS_DIR,
        )
    else:
        return PDFRAG(
            max_chunk_size=max_chunk_size,
            similarity_threshold=0.1,
            project_root=PROJECT_ROOT,
            rag_dir=RAG_DIR,
            results_dir=RESULTS_DIR,
            embeddings_dir_path=EMBEDDINGS_DIR,
        )


def process_pdfs(rag_instance: PDFRAG, max_workers: int = None, batch_size: int = 1000) -> bool:
    """Process PDF files using the LangChain-integrated RAG instance.
    
    Args:
        rag_instance: The PDFRAG instance to use
        max_workers: Number of parallel workers
        batch_size: Batch size for processing chunks
        
    Returns:
        True if processing was successful, False otherwise
    """
    # Find PDF files using the stored RAG directory
    pdf_files = list(rag_instance.rag_dir.glob("*.pdf"))
    
    logger.info(f"Found {len(pdf_files)} PDF files in {rag_instance.rag_dir}")
    for i, pdf_file in enumerate(pdf_files, 1):
        size_mb = pdf_file.stat().st_size / (1024 * 1024)
        logger.info(f"  {i}. {pdf_file.name} ({size_mb:.1f} MB)")
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {rag_instance.rag_dir}")
        logger.info("Add PDF files to the RAG directory:")
        logger.info(f"   cp your_pdfs/*.pdf {rag_instance.rag_dir}/")
        return False
    
    # Process all PDFs using LangChain workflow
    logger.info("Starting parallel processing with LangChain loaders and GPU acceleration...")
    start_time = time.time()
    
    # Use parallel processing with optimized settings
    rag_instance.add_multiple_pdfs(
        [str(f) for f in pdf_files],
        max_workers=max_workers,
        batch_size=batch_size
    )
    
    # Log processing summary
    total_time = time.time() - start_time
    logger.info(f"Total processing time with LangChain workflow: {total_time:.1f} seconds")
    
    if rag_instance.vector_store:
        total_vectors = rag_instance.vector_store.index.ntotal
        logger.info(f"Total vectors: {total_vectors}")
        logger.info(f"Speed: {total_vectors/total_time:.1f} vectors/second")
        return True
    else:
        logger.error(f"Processing failed after {total_time:.1f} seconds")
        return False


def run_qa_session(rag_instance: PDFRAG):
    """Run an interactive Q&A session with the LangChain-integrated RAG system.
    
    Args:
        rag_instance: The PDFRAG instance to use for Q&A
    """
    # Check vector store status
    if rag_instance.vector_store is not None:
        logger.info(f"Vector store loaded with {rag_instance.vector_store.index.ntotal} vectors")
        rag_instance.debug_vector_store()  # Add debug analysis
    else:
        logger.error("Vector store not loaded - cannot run Q&A session")
        return
    
    # Example questions
    questions = [
        "What is the definition of machine learning?",
        "Explain the concept of backpropagation.",
        "What are the main types of neural networks?",
        "How does gradient descent work?",
    ]
    
    for question in questions:
        logger.info("=" * 80)
        logger.info(f"Question: {question}")
        logger.info("-" * 80)
        
        # Show retrieved documents
        logger.info("Retrieved Documents:")
        retrieved = rag_instance.retrieve_documents(question, k=10, score_threshold=0.5)
        for i, doc in enumerate(retrieved):
            logger.info(f"{i+1}. Book: {doc['book']} | Page: {doc['page']} | Score: {doc['score']:.3f}")
            logger.info(f"   Content: {doc['content'][:200]}...")
        
        if not retrieved:
            logger.warning("No documents retrieved! There might be an issue with the embeddings.")
            continue
        
        # Generate answer using LangChain prompt templates
        result = rag_instance.generate_answer(question, score_threshold=0.5)
        
        # Use print for final answer (user-facing output)
        print(f"\n🎯 Answer:\n{result['answer']}\n")
        
        # Use logger for sources (operational information)
        logger.info("Sources:")
        for source, score in zip(result["sources"], result["retrieval_scores"]):
            logger.info(f"- {source} (Score: {score:.3f})")
        logger.info(f"Documents used: {result['context_docs']}")


def run_rebuild_mode(chunk_size: int = 256):
    """Run the system in rebuild mode to rebuild embeddings from scratch using LangChain workflow."""
    logger.info("Fast Embedding Rebuild Mode with LangChain")
    logger.info("=" * 50)
    
    # Initialize RAG system
    rag = create_rag_instance(max_chunk_size=chunk_size)
    
    # Force rebuild
    logger.info("Clearing existing embeddings...")
    rag.rebuild_vector_store()
    
    # Process PDFs with LangChain
    success = process_pdfs(rag)
    
    if success:
        logger.info("Rebuild complete with LangChain workflow!")
        # Quick test
        logger.info("Quick test retrieval:")
        test_results = rag.retrieve_documents("machine learning", k=10)
        for doc in test_results:
            logger.info(f"  {doc['book']} (Page {doc['page']}) - Score: {doc['score']:.3f}")
    else:
        logger.error("Rebuild failed!")


def run_training_mode(
    chunk_size: int = 256, 
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    push_to_hub: bool = False,
    hub_model_id: str = None,
    hub_token: str = None,
    hub_private: bool = False
):
    """Run the system in training mode to train custom embeddings using LangChain loaders.
    
    Args:
        chunk_size: Maximum tokens per chunk
        base_model: Base model to fine-tune
        push_to_hub: Whether to push to Hugging Face Hub
        hub_model_id: Model ID for the Hub
        hub_token: Hugging Face token
        hub_private: Whether to make the repository private
    """
    logger.info("Custom Embedding Training Mode with LangChain Integration")
    logger.info("=" * 50)
    
    # Initialize RAG system
    rag = create_rag_instance(max_chunk_size=chunk_size)
    
    # Check if PDFs exist first
    pdf_files = list(rag.rag_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in {rag.rag_dir}")
        logger.info("Add PDF files to the RAG directory first")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files for training with LangChain loaders")
    if push_to_hub:
        logger.info(f"Training will be pushed to Hub: {hub_model_id or 'auto-generated'}")
        logger.info(f"Repository will be {'private' if hub_private else 'public'}")
    
    # Train custom embedding model with hub parameters
    success = rag.train_custom_embedding_model(
        base_model=base_model,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        hub_token=hub_token,
        hub_private=hub_private
    )
    
    if success:
        logger.info("Embedding training completed with LangChain workflow!")
        if push_to_hub and hub_model_id:
            logger.info(f"Model successfully uploaded to: https://huggingface.co/{hub_model_id}")
        logger.info("Now run with --mode custom to use the custom model:")
        logger.info("   python main_rag.py --mode custom")
    else:
        logger.error("Embedding training failed!")


def run_custom_mode(chunk_size: int = 256, max_workers: int = None, batch_size: int = 1000):
    """Run the system using custom-trained embeddings with LangChain integration."""
    logger.info("Custom Embedding Mode with LangChain")
    logger.info("=" * 50)
    
    # Initialize RAG system with custom embeddings
    rag = create_rag_instance(use_custom_embedding=True, max_chunk_size=chunk_size)
    
    # Process PDFs
    success = process_pdfs(rag, max_workers=max_workers, batch_size=batch_size)
    
    if success:
        # Run Q&A session
        run_qa_session(rag)
    else:
        logger.error("Failed to process PDFs")


def run_normal_mode(chunk_size: int = 256, max_workers: int = None, batch_size: int = 1000):
    """Run the system in normal processing mode with LangChain integration."""
    logger.info("Normal Processing Mode with LangChain")
    logger.info("=" * 50)
    
    # Initialize RAG system
    rag = create_rag_instance(max_chunk_size=chunk_size)
    
    # Process PDFs
    success = process_pdfs(rag, max_workers=max_workers, batch_size=batch_size)
    
    if success:
        # Run Q&A session
        run_qa_session(rag)
    else:
        logger.error("Failed to process PDFs")


def run_streamlit_mode(use_custom: bool = False, chunk_size: int = 256, max_workers: int = None, batch_size: int = 1000):
    """Run the system in Streamlit chat mode with LangChain integration.
    
    Args:
        use_custom: Whether to use custom embeddings
        chunk_size: Maximum tokens per chunk
        max_workers: Number of parallel workers
        batch_size: Batch size for processing
    """
    logger.info("Starting Streamlit Chat Interface with LangChain")
    logger.info("=" * 50)
    
    # Check if custom model exists when requested
    if use_custom:
        custom_model_path = EMBEDDINGS_DIR / "custom_model"
        if not custom_model_path.exists():
            logger.error(f"Custom model requested but not found at {custom_model_path}")
            logger.info("Train a custom model first with: python main_rag.py --mode train")
            return
        logger.info("Using custom embeddings for Streamlit interface with LangChain")
    else:
        logger.info("Using standard embeddings for Streamlit interface with LangChain")
    
    # Initialize RAG system
    rag = create_rag_instance(use_custom_embedding=use_custom, max_chunk_size=chunk_size)
    
    # Process PDFs
    success = process_pdfs(rag, max_workers=max_workers, batch_size=batch_size)
    
    if success:
        # Start Streamlit chat interface
        rag.streamlit_chat()
    else:
        logger.error("Failed to process PDFs - cannot start Streamlit interface")


def run_export_mode(
    chunk_size: int = 256,
    repo_id: str = None,
    hf_filename: str = "rag_document_chunks.jsonl",
    repo_type: str = "dataset",
    private: bool = True,
    hub_token: str = None
):
    """Run the system in export mode to save chunks to JSONL and upload to Hugging Face.
    
    Args:
        chunk_size: Maximum tokens per chunk
        repo_id: Hugging Face repository ID (username/repo-name)
        hf_filename: Name for the file in the HF repository
        repo_type: Type of repository ("dataset" or "model")
        private: Whether the repository should be private
        hub_token: Hugging Face token for authentication
    """
    logger.info("Export Mode - Save Chunks to JSONL and Upload to Hugging Face")
    logger.info("=" * 50)
    
    # Initialize RAG system
    rag = create_rag_instance(max_chunk_size=chunk_size)
    
    # Check if PDFs exist first
    pdf_files = list(rag.rag_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in {rag.rag_dir}")
        logger.info("Add PDF files to the RAG directory first")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files for export")
    
    # Export all chunks to JSONL
    logger.info("Exporting chunks to JSONL...")
    jsonl_path = rag.export_all_chunks_to_jsonl()
    
    if not jsonl_path:
        logger.error("Failed to export chunks to JSONL")
        return
    
    logger.info(f"Successfully exported chunks to: {jsonl_path}")
    
    # Upload to Hugging Face if repo_id is provided
    if repo_id:
        logger.info(f"Uploading to Hugging Face repository: {repo_id}")
        success = rag.upload_chunks_to_huggingface(
            local_file_path=jsonl_path,
            repo_id=repo_id,
            hf_filename=hf_filename,
            repo_type=repo_type,
            private=private,
            token=hub_token
        )
        
        if success:
            logger.info(f"Successfully uploaded to: https://huggingface.co/{repo_id}/{hf_filename}")
        else:
            logger.error("Failed to upload to Hugging Face")
    else:
        logger.info("Skipping Hugging Face upload (no repo_id provided)")
        logger.info(f"JSONL file available at: {jsonl_path}")


def create_argument_parser():
    """Create and configure the argument parser with LangChain integration details."""
    parser = argparse.ArgumentParser(
        description="PDF RAG System with LangChain Integration - Process and query PDF documents using LangChain loaders, text splitters, and vector stores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Normal mode with LangChain PyMuPDFLoader and standard embeddings
  %(prog)s --mode rebuild            # Rebuild vector store from scratch using LangChain workflow
  %(prog)s --mode train              # Train custom embeddings with LangChain loaders
  %(prog)s --mode train --push-to-hub --hub-model-id "myusername/my-embedding-model"  # Train and push to Hub
  %(prog)s --mode train --push-to-hub --hub-private  # Train and push to private Hub repo
  %(prog)s --mode custom             # Use custom embeddings with LangChain integration
  %(prog)s --mode streamlit          # Start Streamlit chat interface with LangChain
  %(prog)s --mode streamlit --custom # Streamlit with custom embeddings and LangChain
  %(prog)s --mode export --repo-id "username/rag-docs"  # Export chunks to JSONL and upload to HF

LangChain Components Used:
  - PyMuPDFLoader/PyPDFLoader for fast PDF extraction
  - SentenceTransformersTokenTextSplitter for intelligent chunking
  - HuggingFaceEmbeddings for embedding generation
  - FAISS vector store for similarity search
  - ChatPromptTemplate/PromptTemplate for structured prompts

Note: For pushing to Hugging Face Hub, add your token to .env file:
  HUGGINGFACE_HUB_TOKEN=hf_xxxxxxxxx
  or login via: huggingface-hub login
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["normal", "rebuild", "train", "custom", "streamlit", "export"],
        default="normal",
        help="Operating mode (default: normal)"
    )
    
    parser.add_argument(
        "--custom",
        action="store_true",
        help="Use custom trained embeddings (only applicable to streamlit mode)"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of parallel workers for PDF processing (default: CPU count * 2)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for processing chunks (default: 1000)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Maximum tokens per chunk (default: 256)"
    )
    
    # Training-specific arguments
    parser.add_argument(
        "--base-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Base model to fine-tune (default: sentence-transformers/all-MiniLM-L6-v2)"
    )
    
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push trained model to Hugging Face Hub (only for train mode)"
    )
    
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="Model ID for Hugging Face Hub (e.g., 'username/model-name'). Auto-generated if not provided."
    )
    
    parser.add_argument(
        "--hub-token",
        type=str,
        default=None,
        help="Hugging Face token for authentication (automatically loaded from HUGGINGFACE_HUB_TOKEN or HF_TOKEN env vars if not provided)"
    )
    
    parser.add_argument(
        "--hub-private",
        action="store_true",
        help="Make the Hugging Face repository private (default: public)"
    )
    
    # Export-specific arguments
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Hugging Face repository ID for uploading chunks (e.g., 'username/rag-docs')"
    )
    
    parser.add_argument(
        "--hf-filename",
        type=str,
        default="rag_document_chunks.jsonl",
        help="Name for the file in the Hugging Face repository (default: rag_document_chunks.jsonl)"
    )
    
    parser.add_argument(
        "--repo-type",
        type=str,
        choices=["dataset", "model"],
        default="dataset",
        help="Type of Hugging Face repository (default: dataset)"
    )
    
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the Hugging Face repository private (default: True)"
    )
    
    return parser


def main():
    """Main function to handle different execution modes."""
    logger.info("PDF RAG System Starting...")
    logger.info("Project structure:")
    logger.info(f"   Script: {SCRIPT_DIR}")
    logger.info(f"   Project root: {PROJECT_ROOT}")
    logger.info(f"   RAG directory: {RAG_DIR}")
    logger.info(f"   Results: {RESULTS_DIR}")
    logger.info(f"   Embeddings: {EMBEDDINGS_DIR}")
    
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Route to appropriate mode
    try:
        if args.mode == "rebuild":
            run_rebuild_mode(chunk_size=args.chunk_size)
        elif args.mode == "train":
            run_training_mode(
                chunk_size=args.chunk_size,
                base_model=args.base_model,
                push_to_hub=args.push_to_hub,
                hub_model_id=args.hub_model_id,
                hub_token=args.hub_token,
                hub_private=args.hub_private
            )
        elif args.mode == "custom":
            run_custom_mode(chunk_size=args.chunk_size, max_workers=args.max_workers, batch_size=args.batch_size)
        elif args.mode == "streamlit":
            run_streamlit_mode(use_custom=args.custom, chunk_size=args.chunk_size, max_workers=args.max_workers, batch_size=args.batch_size)
        elif args.mode == "export":
            run_export_mode(
                chunk_size=args.chunk_size,
                repo_id=args.repo_id,
                hf_filename=args.hf_filename,
                repo_type=args.repo_type,
                private=args.private,
                hub_token=args.hub_token
            )
        else:  # normal mode
            run_normal_mode(chunk_size=args.chunk_size, max_workers=args.max_workers, batch_size=args.batch_size)
    except Exception as e:
        logger.error(f"Error in {args.mode} mode: {e}")
        raise


if __name__ == "__main__":
    main()
