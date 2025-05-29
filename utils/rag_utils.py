#!/usr/bin/env python3
"""
RAG utilities for PDF processing and retrieval using LangChain components.
Contains the PDFRAG class and helper functions for document processing.

Performance optimizations included:
- LangChain document loaders and text splitters
- Simplified vector store operations with FAISS
- CUDA-safe multiprocessing configuration
- Optimized memory usage and batch processing
- Structured prompt templates for better LLM interaction

PDF processing is handled by pdf_utils.py module using LangChain loaders.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
import multiprocessing as mp
import logging
import time
import json
from huggingface_hub import HfApi, create_repo

# LangChain core components
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.prompts.chat import HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# LangChain community components
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline

# Traditional components
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Import PDF utilities from the new pdf_utils module
from .pdf_utils import (
    check_pymupdf_installation,
    extract_documents_from_single_pdf,
    extract_documents_from_multiple_pdfs,
    find_rag_directory,
)

logger = logging.getLogger(__name__)


def setup_multiprocessing_for_cuda():
    """Setup multiprocessing to avoid CUDA context issues.
    
    This function sets the multiprocessing start method to 'spawn' if CUDA is available
    and we haven't already set a start method. This prevents CUDA context inheritance
    issues when using ProcessPoolExecutor with CUDA-enabled parent processes.
    """
    try:
        # Only set start method if none has been set yet
        if mp.get_start_method(allow_none=True) is None:
            if torch.cuda.is_available():
                logger.info("CUDA detected: Setting multiprocessing start method to 'spawn' for compatibility")
                mp.set_start_method('spawn')
            else:
                logger.info("No CUDA detected: Using default multiprocessing start method")
    except RuntimeError as e:
        # Start method already set, which is fine
        logger.info(f"Multiprocessing start method already set: {mp.get_start_method()}")


class PDFRAG:
    def __init__(
        self,
        model_name: str = "RikoteMaster/model_openqa",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_chunk_size: int = 256,  # Changed default to 256 tokens for embeddings
        chunk_overlap: int = 50,   # Now in tokens, not characters
        similarity_threshold: float = 0.7,
        embeddings_dir: Optional[str] = None,
        use_custom_embedding: bool = False,
        custom_embedding_path: Optional[str] = None,
        project_root: Optional[Path] = None,
        rag_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
        embeddings_dir_path: Optional[Path] = None,
        chunks_output_dir: Optional[str] = None,  # New parameter
    ):
        """Initialize PDF RAG system using LangChain components.

        Args:
            model_name: HuggingFace model name for the LLM
            embedding_model: HuggingFace model name for embeddings
            max_chunk_size: Maximum number of tokens per chunk (for embeddings)
            chunk_overlap: Overlap between chunks in tokens
            similarity_threshold: Default threshold for document similarity (0-1)
            embeddings_dir: Directory to save/load embeddings (defaults to project structure)
            use_custom_embedding: Whether to use custom trained embedding model
            custom_embedding_path: Path to custom trained embedding model (defaults to project structure)
            project_root: Project root directory
            rag_dir: RAG directory path
            results_dir: Results directory path
            embeddings_dir_path: Embeddings directory path
            chunks_output_dir: Directory to save JSONL files with chunks
        """
        logger.info(f"Loading model from HuggingFace: {model_name}")
        
        # Setup multiprocessing for CUDA compatibility
        setup_multiprocessing_for_cuda()
        
        # Check for performance optimizations
        check_pymupdf_installation()
        
        # Store directory paths
        self.project_root = project_root
        self.results_dir = results_dir
        
        # Store embedding configuration
        self.use_custom_embedding = use_custom_embedding
        self.custom_embedding_path = custom_embedding_path
        self.embedding_model_name = embedding_model
        self.embedding_chunk_size = max_chunk_size
        self.embedding_chunk_overlap = chunk_overlap
        
        # Auto-detect RAG directory if not provided
        detected_rag_dir = find_rag_directory(project_root, rag_dir, results_dir)
        
        # Set up paths with auto-detection and project-relative defaults
        if embeddings_dir is None:
            embeddings_dir = str(embeddings_dir_path)
        if custom_embedding_path is None:
            custom_embedding_path = str(embeddings_dir_path / "custom_model")
            self.custom_embedding_path = custom_embedding_path
            
        logger.info(f"Using embeddings directory: {embeddings_dir}")
        logger.info(f"Using custom embedding path: {custom_embedding_path}")

        # Set up embeddings directory
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        # Store detected RAG directory for later use
        self.rag_dir = detected_rag_dir

        # Initialize embeddings with GPU support using LangChain HuggingFaceEmbeddings
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device} for embeddings")

        # Choose embedding model - simplified using HuggingFaceEmbeddings directly
        if use_custom_embedding and Path(custom_embedding_path).exists():
            logger.info(f"Using custom trained embedding model: {custom_embedding_path}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=custom_embedding_path,
                model_kwargs={"device": device},
                encode_kwargs={
                    "device": device,
                    "batch_size": 32,
                    "truncate_dim": None,
                },
            )
            embedding_model_for_splitter = custom_embedding_path
            logger.info("Custom embedding model loaded successfully via HuggingFaceEmbeddings")
        else:
            if use_custom_embedding:
                logger.warning(f"Custom embedding model not found at {custom_embedding_path}, using default")
            
            # Use default HuggingFace embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={"device": device},
                encode_kwargs={
                    "device": device,
                    "batch_size": 32,
                    "truncate_dim": None,
                },
            )
            embedding_model_for_splitter = embedding_model

        # Initialize SentenceTransformersTokenTextSplitter for document chunking
        logger.info(f"Initializing SentenceTransformersTokenTextSplitter with model: {embedding_model_for_splitter}")
        
        # Common splitter arguments to avoid duplication
        splitter_args = {
            "model_name": embedding_model_for_splitter,
            "chunk_overlap": self.embedding_chunk_overlap,
        }
        
        self.text_splitter_for_embedding = SentenceTransformersTokenTextSplitter(
            tokens_per_chunk=self.embedding_chunk_size,
            **splitter_args
        )

        # Initialize training splitter (may have different chunk size in the future)
        training_chunk_size = max_chunk_size  # Same as embedding for now, but configurable
        logger.info(f"Initializing training text splitter with {training_chunk_size} tokens per chunk")
        self.text_splitter_for_training = SentenceTransformersTokenTextSplitter(
            tokens_per_chunk=training_chunk_size,
            **splitter_args
        )

        # Initialize vector store
        self.vector_store = None

        # Track processed PDFs
        self.processed_pdfs = set()
        self._load_processed_pdfs()

        # Try to load existing vector store
        self._load_vector_store()

        # Set similarity threshold
        self.similarity_threshold = similarity_threshold

        # Initialize LLM components (only once)
        logger.info(f"Initializing LLM: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True, padding_side="right")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        # Create pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=1000,
            do_sample=True,
            temperature=0.8,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Create LangChain LLM
        self.llm = HuggingFacePipeline(pipeline=self.pipe)

        # Initialize structured prompt template using LangChain ChatPromptTemplate
        self.prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="<|im_start|>system\nYou are a helpful assistant specialised in master‑level STEM.\n<|im_end|>"),
            HumanMessagePromptTemplate.from_template(
                """<|im_start|>user
The following is a question about knowledge and skills in advanced master‑level STEM courses.

Question: {question}

Context information:
{context}
<|im_end|> <thinking> Okay, lets think."""
            )
        ])
        
        # For non-chat models, keep a simple template as fallback
        self.simple_prompt_template = PromptTemplate.from_template(
            """<|im_start|>system
You are a helpful assistant specialised in master‑level STEM.
<|im_end|>

<|im_start|>user
The following is a question about knowledge and skills in advanced master‑level STEM courses.

Question: {question}

Context information:
{context}
<|im_end|>

<|im_start|>assistant
<thinking> Okay, lets think."""
        )

        logger.info("Model and components loaded with LangChain integration")
        logger.info(f"RAG system ready with directory: {self.rag_dir}")
        logger.info(f"Using SentenceTransformersTokenTextSplitter with {self.embedding_chunk_size} tokens per chunk")

        # Initialize chunks output directory
        if chunks_output_dir is None:
            chunks_output_dir = str(self.embeddings_dir / "chunks")
        self.chunks_output_dir = Path(chunks_output_dir)
        self.chunks_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Hugging Face API
        self.hf_api = HfApi()
        
        logger.info(f"Using chunks output directory: {self.chunks_output_dir}")

    def _load_vector_store(self):
        """Load existing vector store using LangChain FAISS methods."""
        index_path = self.embeddings_dir / "faiss_index"
        logger.info(f"Checking for vector store at: {index_path}")

        if index_path.exists():
            try:
                logger.info("Loading existing vector store with LangChain FAISS...")
                self.vector_store = FAISS.load_local(
                    str(index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )

                # Check if vector store actually loaded with data
                if (
                    hasattr(self.vector_store, "index")
                    and self.vector_store.index.ntotal > 0
                ):
                    logger.info(
                        f"Vector store loaded successfully with {self.vector_store.index.ntotal} vectors"
                    )
                else:
                    logger.warning("Vector store loaded but appears to be empty")
                    self.vector_store = None

            except Exception as e:
                logger.warning(f"Could not load vector store: {e}")
                logger.warning(f"Error type: {type(e).__name__}")
                import traceback

                traceback.print_exc()
                self.vector_store = None
        else:
            logger.info("No existing vector store found")

    def _save_vector_store(self):
        """Save vector store using LangChain FAISS methods."""
        if self.vector_store is not None:
            try:
                index_path = self.embeddings_dir / "faiss_index"
                logger.info("Saving vector store with LangChain FAISS...")
                self.vector_store.save_local(str(index_path))
                logger.info(
                    f"Vector store saved successfully with {self.vector_store.index.ntotal} vectors"
                )
            except Exception as e:
                logger.warning(f"Could not save vector store: {e}")

    def _load_processed_pdfs(self):
        """Load list of processed PDFs."""
        processed_file = self.embeddings_dir / "processed_pdfs.txt"
        if processed_file.exists():
            with open(processed_file, "r") as f:
                self.processed_pdfs = set(line.strip() for line in f)
            logger.info(f"Loaded {len(self.processed_pdfs)} processed PDFs")

    def _save_processed_pdfs(self):
        """Save list of processed PDFs."""
        processed_file = self.embeddings_dir / "processed_pdfs.txt"
        with open(processed_file, "w") as f:
            for pdf in sorted(self.processed_pdfs):
                f.write(f"{pdf}\n")
        logger.info(f"Saved {len(self.processed_pdfs)} processed PDFs")

    def _save_chunks_to_jsonl(self, chunks: List[Document], pdf_name: str, append: bool = True) -> str:
        """Save chunks to a JSONL file.
        
        Args:
            chunks: List of Document objects to save
            pdf_name: Name of the source PDF
            append: Whether to append to existing file or create new
            
        Returns:
            Path to the saved JSONL file
        """
        output_file = self.chunks_output_dir / "rag_chunks.jsonl"
        mode = "a" if append and output_file.exists() else "w"
        
        with open(output_file, mode, encoding="utf-8") as f:
            for chunk in chunks:
                # Convert Document to dict format with source at top level
                chunk_dict = {
                    "text": chunk.page_content,
                    "source": pdf_name,  # Moved to top level for lighteval compatibility
                    "page": chunk.metadata.get("page", "Unknown"),
                    "book": chunk.metadata.get("book", "Unknown"),
                    "chunk_index": chunk.metadata.get("chunk_index", 0)
                }
                f.write(json.dumps(chunk_dict, ensure_ascii=False) + "\n")
        
        logger.info(f"Saved {len(chunks)} chunks from {pdf_name} to {output_file}")
        return str(output_file)

    def export_all_chunks_to_jsonl(self, output_filepath: Optional[str] = None) -> str:
        """Export all chunks from processed PDFs to a single JSONL file.
        
        Args:
            output_filepath: Optional custom path for the output file
            
        Returns:
            Path to the generated JSONL file
        """
        if output_filepath is None:
            output_filepath = str(self.chunks_output_dir / "all_rag_chunks.jsonl")
        
        # Get all PDFs from rag_dir
        pdf_paths = list(self.rag_dir.glob("*.pdf"))
        if not pdf_paths:
            logger.error(f"No PDF files found in {self.rag_dir}")
            return ""
        
        logger.info(f"Processing {len(pdf_paths)} PDFs for JSONL export")
        
        # Process each PDF and collect chunks
        all_chunks = []
        for pdf_path in pdf_paths:
            try:
                # Extract documents
                documents = extract_documents_from_single_pdf(pdf_path)
                if not documents:
                    continue
                
                # Split into chunks
                chunks = self.text_splitter_for_embedding.split_documents(documents)
                all_chunks.extend(chunks)
                
                logger.info(f"Processed {pdf_path.name}: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_path.name}: {e}")
                continue
        
        if not all_chunks:
            logger.error("No chunks generated from PDFs")
            return ""
        
        # Save all chunks to JSONL with source at top level
        with open(output_filepath, "w", encoding="utf-8") as f:
            for chunk in all_chunks:
                chunk_dict = {
                    "text": chunk.page_content,
                    "source": chunk.metadata.get("source", "Unknown"),  # Moved to top level for lighteval compatibility
                    "page": chunk.metadata.get("page", "Unknown"),
                    "book": chunk.metadata.get("book", "Unknown"),
                    "chunk_index": chunk.metadata.get("chunk_index", 0)
                }
                f.write(json.dumps(chunk_dict, ensure_ascii=False) + "\n")
        
        logger.info(f"Exported {len(all_chunks)} chunks to {output_filepath}")
        return output_filepath

    def upload_chunks_to_huggingface(
        self,
        local_file_path: str,
        repo_id: str,
        hf_filename: str = "rag_document_chunks.jsonl",
        repo_type: str = "dataset",
        private: bool = True,
        token: Optional[str] = None
    ) -> bool:
        """Upload chunks JSONL file to Hugging Face.
        
        Args:
            local_file_path: Path to the local JSONL file
            repo_id: Hugging Face repository ID (username/repo-name)
            hf_filename: Name for the file in the HF repository
            repo_type: Type of repository ("dataset" or "model")
            private: Whether the repository should be private
            token: Hugging Face API token (optional if logged in)
            
        Returns:
            bool: True if upload was successful
        """
        try:
            # Create repository if it doesn't exist
            try:
                create_repo(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    private=private,
                    token=token,
                    exist_ok=True
                )
                logger.info(f"Repository {repo_id} is ready")
            except Exception as e:
                logger.error(f"Error creating repository {repo_id}: {e}")
                return False
            
            # Upload the file
            self.hf_api.upload_file(
                path_or_fileobj=local_file_path,
                path_in_repo=hf_filename,
                repo_id=repo_id,
                repo_type=repo_type,
                token=token
            )
            
            logger.info(f"Successfully uploaded {local_file_path} to {repo_id}/{hf_filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading to Hugging Face: {e}")
            return False

    def add_pdf(self, pdf_path: str):
        """Add a PDF book to the vector store using LangChain document processing pipeline.

        Args:
            pdf_path: Path to the PDF file
        """
        # Check if PDF was already processed
        pdf_name = Path(pdf_path).name
        if pdf_name in self.processed_pdfs:
            logger.info(f"Skipping already processed PDF: {pdf_name}")
            return

        logger.info(f"Processing PDF with LangChain pipeline: {pdf_path}")

        try:
            # Step 1: Extract documents using LangChain loader
            documents = extract_documents_from_single_pdf(Path(pdf_path))

            if not documents:
                logger.warning(f"No valid documents found in {pdf_name}")
                return

            # Step 2: Chunk documents using LangChain splitter
            chunks = self.text_splitter_for_embedding.split_documents(documents)

            if not chunks:
                logger.warning(f"No valid chunks found in {pdf_name}")
                return

            # Save chunks to JSONL
            self._save_chunks_to_jsonl(chunks, pdf_name)

            # Step 3: Create or update vector store
            if self.vector_store is None:
                logger.info("Creating new vector store with LangChain FAISS...")
                self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            else:
                logger.info("Adding to existing vector store with LangChain FAISS...")
                self.vector_store.add_documents(chunks)

            logger.info(f"Added {len(documents)} pages ({len(chunks)} chunks) from {Path(pdf_path).stem}")

            # Mark PDF as processed and save
            self.processed_pdfs.add(pdf_name)
            self._save_processed_pdfs()

            # Save updated vector store
            self._save_vector_store()

        except Exception as e:
            logger.error(f"Error processing {pdf_name}: {str(e)}")
            raise

    def retrieve_documents(
        self,
        query: str,
        k: int = 10,
        score_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        """Retrieve relevant documents with scores and metadata using LangChain FAISS.

        Args:
            query: The search query
            k: Number of documents to retrieve
            score_threshold: Optional minimum similarity score (0-1)
            filter_metadata: Optional metadata filters to apply

        Returns:
            List of dictionaries containing document content, score, and metadata
        """
        logger.info(
            f"Retrieving documents for query: '{query[:50]}...' (vector_store exists: {self.vector_store is not None})"
        )

        if self.vector_store is None:
            logger.error("No vector store available for retrieval")
            return []

        try:
            # Use LangChain FAISS similarity search with scores
            docs_and_scores = self.vector_store.similarity_search_with_score(
                query, k=k, filter=filter_metadata
            )

            logger.info(f"Found {len(docs_and_scores)} documents before filtering")

            # Filter by score threshold if specified
            if score_threshold is not None:
                docs_and_scores = [
                    (doc, score)
                    for doc, score in docs_and_scores
                    if score >= score_threshold
                ]
                logger.info(
                    f"INFO: {len(docs_and_scores)} documents after score filtering (threshold: {score_threshold})"
                )

            # Format results - working with LangChain Document objects
            results = []
            for doc, score in docs_and_scores:
                results.append(
                    {
                        "content": doc.page_content,
                        "book": doc.metadata.get("book", "Unknown"),
                        "page": doc.metadata.get("page", "Unknown"),
                        "score": float(score),
                    }
                )

            return results

        except Exception as e:
            logger.error(f"Error during document retrieval: {e}")
            import traceback

            traceback.print_exc()
            return []

    def generate_answer(
        self,
        question: str,
        k: int = 10,
        score_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate an answer based on retrieved documents using LangChain RAG chain.
        
        Args:
            question: The question to answer
            k: Number of documents to retrieve
            score_threshold: Optional minimum similarity score
            filter_metadata: Optional metadata filters
            
        Returns:
            Dictionary containing answer, sources, and scores
        """
        if self.vector_store is None:
            logger.error("No vector store available for retrieval")
            return {
                "answer": "Error: No vector store available. Please add some PDFs first.",
                "sources": [],
                "retrieval_scores": [],
                "context_docs": 0
            }

        try:
            # Create retriever from vector store
            retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    'k': k,
                    'score_threshold': score_threshold or self.similarity_threshold,
                    'filter': filter_metadata
                }
            )
            
            # Format documents for context
            def format_docs(docs):
                return "\n\n".join(
                    f"From {doc.metadata.get('book', 'Unknown')} (Page {doc.metadata.get('page', 'Unknown')}):\n{doc.page_content}"
                    for doc in docs
                )
            
            # Create answer generation chain that expects context and question
            answer_generation_chain = (
                self.prompt_template
                | self.llm
                | StrOutputParser()
            )
            
            # Create complete RAG chain with proper data flow
            rag_chain_with_sources = RunnableParallel(
                {
                    "context_docs": retriever,  # Raw documents from retriever
                    "question": RunnablePassthrough()  # Original question
                }
            ).assign(
                # Format retrieved documents for context
                context=lambda x: format_docs(x["context_docs"])
            ).assign(
                # Generate answer using formatted context and question
                answer=answer_generation_chain
            )
            
            # Get answer and sources
            result = rag_chain_with_sources.invoke(question)
            
            # Format sources
            sources = []
            for doc in result["context_docs"]:
                sources.append(f"{doc.metadata.get('book', 'Unknown')} (Page {doc.metadata.get('page', 'Unknown')})")
            
            return {
                "answer": result["answer"],
                "sources": sources,
                "retrieval_scores": [doc.metadata.get("score", 0.0) for doc in result["context_docs"]],
                "context_docs": len(result["context_docs"])
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            import traceback
            traceback.print_exc()
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
                "retrieval_scores": [],
                "context_docs": 0
            }

    def debug_vector_store(self):
        """Debug function to analyze vector store contents."""
        if self.vector_store is None:
            logger.error("No vector store to debug")
            return

        logger.info("Vector Store Debug Information:")
        logger.info(f"Total vectors: {self.vector_store.index.ntotal}")

        # Get a sample of documents to see what's in the store
        try:
            # Get all document metadata
            docstore = self.vector_store.docstore
            logger.info(f"Docstore size: {len(docstore._dict)}")

            # Count documents by book
            book_counts = {}
            for doc_id, doc in docstore._dict.items():
                book_name = doc.metadata.get("book", "Unknown")
                book_counts[book_name] = book_counts.get(book_name, 0) + 1

            logger.info("Documents by book:")
            for book, count in sorted(book_counts.items()):
                logger.info(f"  {book}: {count} chunks")

            # Test embedding quality with a simple query
            logger.info("Testing embedding quality:")
            test_queries = [
                "machine learning",
                "neural networks",
                "backpropagation",
                "gradient descent",
            ]

            for query in test_queries:
                docs = self.vector_store.similarity_search_with_score(query, k=10)
                logger.info(f"\nQuery: '{query}'")
                for i, (doc, score) in enumerate(docs):
                    book = doc.metadata.get("book", "Unknown")
                    page = doc.metadata.get("page", "Unknown")
                    content_preview = doc.page_content[:100].replace("\n", " ")
                    logger.info(f"  {i+1}. {book} (Page {page}) - Score: {score:.3f}")
                    logger.info(f"     Content: {content_preview}...")

        except Exception as e:
            logger.error(f"Error during debug: {e}")
            import traceback

            traceback.print_exc()

    def rebuild_vector_store(self):
        """Rebuild the vector store from scratch."""
        logger.info("Rebuilding vector store from scratch...")

        # Clear existing data
        self.vector_store = None
        self.processed_pdfs = set()

        # Remove existing files
        try:
            import shutil

            if (self.embeddings_dir / "faiss_index").exists():
                shutil.rmtree(self.embeddings_dir / "faiss_index")
            if (self.embeddings_dir / "processed_pdfs.txt").exists():
                os.remove(self.embeddings_dir / "processed_pdfs.txt")
            logger.info("Cleared existing vector store")
        except Exception as e:
            logger.warning(f"Error clearing existing data: {e}")

    def _filter_unprocessed_pdfs(self, pdf_paths: List[str]) -> List[str]:
        """Filter out already processed PDFs.
        
        Args:
            pdf_paths: List of PDF file paths to filter
            
        Returns:
            List of unprocessed PDF paths
        """
        unprocessed_pdfs = []
        for pdf_path in pdf_paths:
            pdf_name = Path(pdf_path).name
            if pdf_name not in self.processed_pdfs:
                unprocessed_pdfs.append(pdf_path)
            else:
                logger.info(f"Skipping already processed: {pdf_name}")
        
        return unprocessed_pdfs

    def add_multiple_pdfs(self, pdf_paths: List[str], max_workers: int = None, batch_size: int = 32, use_parallel: bool = True):
        """Add multiple PDFs efficiently using LangChain document processing pipeline.
        
        This method provides a unified interface for processing multiple PDFs:
        - Parallel processing (default): Fast extraction using multiple workers
        - Sequential processing (fallback): Reliable single-threaded processing
        - CUDA-safe: All chunking happens in main process
        - Efficient batching for vector store operations using LangChain

        Args:
            pdf_paths: List of PDF file paths
            max_workers: Number of parallel workers (defaults to CPU count)
            batch_size: Number of documents to process in each batch
            use_parallel: Whether to use parallel processing (default: True)
        """
        logger.info(f"Processing {len(pdf_paths)} PDFs with LangChain pipeline (parallel={use_parallel}, batch_size={batch_size})")

        # Filter out already processed PDFs
        unprocessed_pdfs = self._filter_unprocessed_pdfs(pdf_paths)

        if not unprocessed_pdfs:
            logger.info("All PDFs already processed!")
            return

        logger.info(f"Processing {len(unprocessed_pdfs)} new PDFs...")
        start_time = time.time()

        # Step 1: Extract documents from all PDFs
        if use_parallel:
            logger.info(f"🚀 Using parallel extraction with LangChain loaders ({max_workers or 'auto'} workers)")
            try:
                pdf_paths_as_path_objects = [Path(p) for p in unprocessed_pdfs]
                all_documents = extract_documents_from_multiple_pdfs(
                    pdf_paths_as_path_objects, max_workers=max_workers
                )
            except Exception as e:
                logger.warning(f"Parallel processing failed: {e}. Falling back to sequential...")
                use_parallel = False
        
        if not use_parallel:
            logger.info("📝 Using sequential extraction with LangChain loaders")
            all_documents = []
            for pdf_path in unprocessed_pdfs:
                try:
                    documents = extract_documents_from_single_pdf(Path(pdf_path))
                    all_documents.extend(documents)
                    logger.info(f"Extracted {len(documents)} documents from {Path(pdf_path).name}")
                except Exception as e:
                    logger.error(f"Error extracting from {Path(pdf_path).name}: {e}")
                    continue

        if not all_documents:
            logger.warning("No valid documents extracted!")
            return

        logger.info(f"✅ Extracted {len(all_documents)} documents total")

        # Step 2: Chunk all documents using LangChain splitter
        logger.info(f"📝 Chunking {len(all_documents)} documents with LangChain splitter...")
        chunks = self.text_splitter_for_embedding.split_documents(all_documents)

        if not chunks:
            logger.warning("No valid chunks created!")
            return

        logger.info(f"Created {len(chunks)} chunks")

        # Step 3: Add chunks to vector store in batches using LangChain methods
        logger.info(f"💾 Adding {len(chunks)} chunks to vector store (batch_size={batch_size})")

        try:
            # Process chunks in batches
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                
                # Save batch chunks to JSONL
                for chunk in batch_chunks:
                    pdf_name = chunk.metadata.get("source", "Unknown")
                    self._save_chunks_to_jsonl([chunk], pdf_name)
                
                if self.vector_store is None:
                    logger.info("Creating new vector store with LangChain FAISS...")
                    self.vector_store = FAISS.from_documents(batch_chunks, self.embeddings)
                else:
                    batch_num = i//batch_size + 1
                    total_batches = (len(chunks) + batch_size - 1)//batch_size
                    logger.info(f"Adding batch {batch_num}/{total_batches} with LangChain FAISS...")
                    self.vector_store.add_documents(batch_chunks)

            # Mark all PDFs as processed
            processed_files = [Path(pdf_path).name for pdf_path in unprocessed_pdfs]
            for pdf_name in processed_files:
                self.processed_pdfs.add(pdf_name)

            self._save_processed_pdfs()
            self._save_vector_store()

            total_time = time.time() - start_time
            mode_str = "parallel" if use_parallel else "sequential"
            logger.info(f"🎉 Successfully processed {len(processed_files)} PDFs with {len(chunks)} chunks using LangChain")
            logger.info(f"⏱️  Total time: {total_time:.2f}s | Speed: {len(chunks)/total_time:.2f} chunks/s ({mode_str})")

        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}")
            raise

    def _prepare_training_corpus_from_pdfs(self) -> List[str]:
        """Prepare training corpus from PDF files using LangChain document processing pipeline.
        
        Returns:
            List of text chunks suitable for training
        """
        pdf_paths = list(self.rag_dir.glob("*.pdf"))
        if not pdf_paths:
            logger.error(f"No PDF files found in {self.rag_dir}")
            return []
        
        logger.info(f"Found {len(pdf_paths)} PDF files to process for training")
        
        # Step 1: Extract documents using LangChain loaders
        all_documents = extract_documents_from_multiple_pdfs(pdf_paths, max_workers=None)

        if not all_documents:
            logger.warning("No documents extracted from PDFs for training")
            return []

        logger.info(f"Extracted {len(all_documents)} documents for training")

        # Step 2: Chunk documents using LangChain splitter
        training_chunks = []
        for doc in all_documents:
            try:
                # Split each document's content
                text_chunks = self.text_splitter_for_training.split_text(doc.page_content)
                for chunk_text in text_chunks:
                    if chunk_text.strip() and len(chunk_text.strip()) >= 50:  # Filter short chunks
                        training_chunks.append(chunk_text.strip())
            except Exception as e:
                logger.warning(f"Error chunking document: {e}")
                continue

        logger.info(f"Prepared {len(training_chunks)} training chunks using LangChain pipeline")
        return training_chunks

    def streamlit_chat(self):
        """Set up an interactive chat interface using Streamlit."""
        # Import Streamlit dependencies only when needed
        import streamlit as st
        from streamlit_chat import message
        import asyncio
        import nest_asyncio

        # Apply nest_asyncio to allow nested event loops
        nest_asyncio.apply()

        # Initialize session state for chat history and sources
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "sources" not in st.session_state:
            st.session_state.sources = {}
        if "initialized" not in st.session_state:
            st.session_state.initialized = False

        # Set up the page
        st.set_page_config(
            page_title="PDF RAG Chat",
            page_icon="📚",
            layout="wide"
        )

        st.title("📚 PDF RAG Chat")
        st.markdown("""
        <style>
        .stChatMessage { margin-bottom: 1.5em; }
        .stButton>button { margin-top: 0.5em; }
        .st-collapsed, .st-collapsible { background: #222 !important; color: #fff !important; }
        </style>
        """, unsafe_allow_html=True)
        st.markdown("""
        This chat interface allows you to ask questions about the content of your PDF documents.<br>
        The system will retrieve relevant information and generate answers based on the context.
        """, unsafe_allow_html=True)

        # Initialize the system only once
        if not st.session_state.initialized:
            with st.spinner("Initializing system..."):
                if self.vector_store is None:
                    self._load_vector_store()
                test_query = "test"
                try:
                    self.retrieve_documents(test_query, k=1)
                    st.session_state.initialized = True
                    st.success("System initialized successfully!")
                except Exception as e:
                    st.error(f"Error initializing system: {str(e)}")
                    return

        # Display chat history
        for i, message_obj in enumerate(st.session_state.messages):
            role = message_obj["role"]
            content = message_obj["content"]
            sources = message_obj.get("sources", None)
            with st.chat_message(role):
                st.markdown(content)
                # Only for assistant, show sources if present
                if role == "assistant" and sources:
                    with st.expander("Show Sources", expanded=False):
                        for src in sources:
                            st.markdown(src)

        # Chat input
        prompt = st.chat_input("Ask a question about your PDFs")
        if prompt:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = self.generate_answer(prompt, score_threshold=0.5)
                        response = result["answer"]
                        sources = []
                        for source, score in zip(result["sources"], result["retrieval_scores"]):
                            sources.append(f"- {source} (Score: {score:.3f})")
                        # Add assistant response to chat history, with sources
                        st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})
                        st.markdown(response)
                        if sources:
                            with st.expander("Show Sources", expanded=False):
                                for src in sources:
                                    st.markdown(src)
                    except Exception as e:
                        response = f"An error occurred: {str(e)}"
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.error(response)

        # Add a sidebar with information
        with st.sidebar:
            st.title("About")
            st.markdown("""
            This chat interface uses:
            - PDF RAG system for document retrieval
            - Advanced language model for answer generation
            - Semantic search with similarity scoring
            """)
            if self.vector_store is not None:
                st.success(f"✅ Vector store loaded with {self.vector_store.index.ntotal} vectors")
            else:
                st.error("❌ Vector store not loaded")
            if self.processed_pdfs:
                st.subheader("Processed PDFs")
                for pdf in sorted(self.processed_pdfs):
                    st.text(f"📄 {pdf}")

    def train_custom_embedding_model(
        self, 
        base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        push_to_hub: bool = False,
        hub_model_id: Optional[str] = None,
        hub_token: Optional[str] = None,
        hub_private: bool = False
    ) -> bool:
        """Train a custom embedding model using the PDF corpus with SentenceTransformers v3+ API and LangChain loaders.
        
        Args:
            base_model: Base model to fine-tune
            push_to_hub: Whether to push the trained model to Hugging Face Hub
            hub_model_id: Model ID for the Hub (e.g., "username/model-name")
            hub_token: Hugging Face token for authentication (optional if logged in)
            hub_private: Whether to make the repository private
        """
        try:
            # Import required libraries for training (only when needed)
            from sentence_transformers import SentenceTransformer
            from sentence_transformers.losses import MultipleNegativesRankingLoss
            from sentence_transformers.training_args import SentenceTransformerTrainingArguments
            from sentence_transformers.trainer import SentenceTransformerTrainer
            from sentence_transformers.readers.InputExample import InputExample
            import json
            import numpy as np
            
            # Load environment variables to access tokens
            try:
                from dotenv import load_dotenv
                load_dotenv()
                logger.info("Environment variables loaded for Hub authentication")
            except ImportError:
                logger.warning("python-dotenv not available, using system environment variables only")
            
            # Use project-relative paths
            pdf_dir = str(self.rag_dir)
            output_dir = str(self.embeddings_dir / "custom_model")
            
            logger.info("Training custom embedding model with SentenceTransformers v3+ API and LangChain loaders...")
            logger.info(f"PDF directory: {pdf_dir}")
            logger.info(f"Output directory: {output_dir}")
            logger.info(f"Base model: {base_model}")
            
            # Set device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            # Create output directory
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            
            # 1. Load base model
            logger.info(f"Loading base model: {base_model}")
            model = SentenceTransformer(base_model, device=device)
            
            # 2. Prepare training corpus from PDFs using LangChain workflow
            logger.info("Preparing training corpus from PDFs with LangChain loaders...")
            texts = self._prepare_training_corpus_from_pdfs()
            
            if not texts:
                logger.error("No valid text chunks extracted for training")
                return False
            
            logger.info(f"Training corpus size: {len(texts)} chunks")
            
            # 3. Create training examples with noise function (TSDAE approach)
            logger.info("Creating TSDAE training examples with noise function...")
            
            def delete_words(text, del_ratio=0.6):
                """Delete words from text for denoising training."""
                try:
                    import nltk
                    from nltk import word_tokenize
                    from nltk.tokenize.treebank import TreebankWordDetokenizer
                    
                    # Download required NLTK data if not available
                    try:
                        nltk.data.find('tokenizers/punkt')
                    except LookupError:
                        nltk.download('punkt')
                    
                    words = word_tokenize(text)
                    n = len(words)
                    if n == 0:
                        return text

                    keep_or_not = np.random.rand(n) > del_ratio
                    if sum(keep_or_not) == 0:
                        keep_or_not[np.random.choice(n)] = True  # guarantee that at least one word remains
                    
                    words_processed = TreebankWordDetokenizer().detokenize(np.array(words)[keep_or_not])
                    return words_processed
                except ImportError:
                    # Fallback: simple word deletion without NLTK
                    words = text.split()
                    n = len(words)
                    if n == 0:
                        return text
                    
                    keep_or_not = np.random.rand(n) > del_ratio
                    if sum(keep_or_not) == 0:
                        keep_or_not[np.random.choice(n)] = True
                    
                    return ' '.join(np.array(words)[keep_or_not])
            
            # Create training examples: [noisy_text, original_text]
            train_examples = []
            for text in texts:
                noisy_text = delete_words(text)
                train_examples.append(InputExample(texts=[noisy_text, text]))
            
            logger.info(f"Created {len(train_examples)} training examples")
            
            # Convert to dataset format expected by SentenceTransformerTrainer
            from datasets import Dataset as HFDataset
            
            # Prepare data in the format expected by the trainer
            train_data = {
                "anchor": [],
                "positive": []
            }
            
            for example in train_examples:
                train_data["anchor"].append(example.texts[0])  # noisy text
                train_data["positive"].append(example.texts[1])  # original text
            
            # Split data into train/validation to prevent overfitting
            total_examples = len(train_examples)
            validation_split = 0.1  # Use 10% for validation
            split_idx = int(total_examples * (1 - validation_split))
            
            train_split_data = {
                "anchor": train_data["anchor"][:split_idx],
                "positive": train_data["positive"][:split_idx]
            }
            
            val_split_data = {
                "anchor": train_data["anchor"][split_idx:],
                "positive": train_data["positive"][split_idx:]
            }
            
            train_dataset = HFDataset.from_dict(train_split_data)
            val_dataset = HFDataset.from_dict(val_split_data)
            
            logger.info(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation examples")
            logger.info(f"Validation split ratio: {validation_split:.1%}")
            
            # 4. Define the TSDAE loss
            logger.info("Setting up TSDAE loss function...")
            
            # For the new API, we need to use a loss that works with the dataset format
            # Let's use MultipleNegativesRankingLoss which works well for this type of training
            loss = MultipleNegativesRankingLoss(model=model)
            
            # 5. Configure training arguments
            epochs = 5
            batch_size = 128
            learning_rate = 2e-5
            warmup_ratio = 0.1
            
            logger.info(f"Training configuration:")
            logger.info(f"  Epochs: {epochs}")
            logger.info(f"  Batch size: {batch_size}")
            logger.info(f"  Learning rate: {learning_rate}")
            logger.info(f"  Warmup ratio: {warmup_ratio}")
            if push_to_hub:
                logger.info(f"  Push to Hub: {hub_model_id} (private: {hub_private})")
            
            # Configure hub settings if push_to_hub is enabled
            hub_kwargs = {}
            if push_to_hub:
                if hub_model_id is None:
                    # Generate a default model ID based on base model and timestamp
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    base_name = base_model.split("/")[-1] if "/" in base_model else base_model
                    hub_model_id = f"custom-embedding-{base_name}-{timestamp}"
                    logger.info(f"Generated hub model ID: {hub_model_id}")
                
                # Use token from environment if not provided
                if hub_token is None:
                    hub_token = os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HF_TOKEN')
                    if hub_token:
                        logger.info("Using Hugging Face token from environment variables")
                    else:
                        logger.info("No token provided - assuming you're logged in via `huggingface-hub login`")
                
                hub_kwargs = {
                    "push_to_hub": True,
                    "hub_model_id": hub_model_id,
                    "hub_strategy": "end",  # Push at the end of training
                    "hub_private_repo": hub_private,
                }
                
                if hub_token:
                    hub_kwargs["hub_token"] = hub_token
            
            training_args = SentenceTransformerTrainingArguments(
                output_dir=str(output_dir_path),
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                warmup_ratio=warmup_ratio,
                # Evaluation configuration (simplified for SentenceTransformerTrainingArguments)
                eval_strategy="steps",  # Use eval_strategy instead of evaluation_strategy
                eval_steps=200,  # Evaluate every 200 training steps
                save_strategy="steps",
                save_steps=200,  # Save every 200 steps (aligned with eval)
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,  # Lower loss is better
                # Logging and saving
                logging_steps=50,
                save_total_limit=3,  # Keep only 3 best checkpoints
                dataloader_drop_last=True,
                # Additional settings for better training
                fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
                dataloader_num_workers=2,  # Parallel data loading
                # Hub configuration
                **hub_kwargs,
            )
            
            # 6. Create trainer and start training
            logger.info("Initializing SentenceTransformerTrainer...")
            
            # Import EarlyStoppingCallback for better overfitting control
            from transformers import EarlyStoppingCallback
            
            trainer = SentenceTransformerTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,  # Add validation dataset
                loss=loss,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],  # Early stopping
            )
            
            logger.info("Starting training with evaluation and early stopping...")
            logger.info(f"  Training examples: {len(train_dataset)}")
            logger.info(f"  Validation examples: {len(val_dataset)}")
            logger.info(f"  Early stopping patience: 5 evaluations")
            logger.info(f"  Evaluation every {training_args.eval_steps} steps")
            
            trainer.train()
            
            # 7. Save the final model
            logger.info("Saving trained model...")
            trainer.save_model()
            
            logger.info(f"Custom embedding model training completed!")
            logger.info(f"Model saved to: {output_dir_path}")
            
            # 8. Save additional metadata
            metadata = {
                "base_model": base_model,
                "training_corpus_size": len(texts),
                "training_examples": len(train_examples),
                "train_examples": len(train_dataset),
                "validation_examples": len(val_dataset),
                "validation_split": validation_split,
                "chunk_size": 256,
                "chunk_overlap": 50,
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "warmup_ratio": warmup_ratio,
                "early_stopping_patience": 5,
                "eval_steps": training_args.eval_steps,
                "device": device,
                "training_api": "SentenceTransformers_v3+_with_LangChain",
                "loss_function": "MultipleNegativesRankingLoss",
                "eval_strategy": "steps",
                "fp16": torch.cuda.is_available(),
                "save_strategy": "steps",
                "save_steps": training_args.save_steps,
                "logging_steps": training_args.logging_steps,
                "save_total_limit": training_args.save_total_limit,
                # Hub configuration
                "push_to_hub": push_to_hub,
                "hub_model_id": hub_model_id if push_to_hub else None,
                "hub_private": hub_private if push_to_hub else None,
            }
            
            metadata_path = output_dir_path / "training_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Training metadata saved to: {metadata_path}")
            logger.info("Custom embedding model trained successfully with LangChain workflow!")
            
            if push_to_hub and hub_model_id:
                logger.info(f"🤗 Model uploaded to Hugging Face Hub: https://huggingface.co/{hub_model_id}")
            
            logger.info("To use it, initialize PDFRAG with:")
            logger.info(f"   use_custom_embedding=True")
            logger.info(f"   custom_embedding_path='{output_dir}'")

            return True
                
        except ImportError as e:
            logger.error(f"Could not import required libraries: {e}")
            logger.info("Install required packages: pip install sentence-transformers nltk")
            return False
        except Exception as e:
            logger.error(f"Error training custom embedding model: {e}")
            import traceback
            traceback.print_exc()
            return False 