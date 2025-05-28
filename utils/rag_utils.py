#!/usr/bin/env python3
"""
RAG utilities for PDF processing and retrieval.
Contains the PDFRAG class and helper functions for document processing.

Performance optimizations included:
- Modular design with PDF utilities separated into pdf_utils.py
- CUDA-safe multiprocessing configuration
- Optimized memory usage and batch processing
- Fast vector store operations for large document collections

PDF processing is now handled by pdf_utils.py module.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Any
import multiprocessing as mp
import logging
import time

from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Import PDF utilities from the new pdf_utils module
from .pdf_utils import (
    check_pymupdf_installation,
    extract_text_from_single_pdf,
    extract_text_from_multiple_pdfs,
    find_rag_directory,
)

logger = logging.getLogger(__name__)


class CustomEmbeddingWrapper:
    """Wrapper to make custom SentenceTransformer models compatible with LangChain."""
    
    def __init__(self, model):
        self.model = model
        # Use the more robust method to get max sequence length
        try:
            self.max_seq_length = model.get_max_seq_length()
        except (AttributeError, Exception):
            # Fallback to the previous method if get_max_seq_length() is not available
            self.max_seq_length = getattr(model, 'max_seq_length', 256)
        
    def embed_documents(self, texts):
        # Let SentenceTransformer handle truncation internally
        # The encode method automatically handles texts that exceed max_seq_length
        return self.model.encode(texts, truncate_dim=None).tolist()
        
    def embed_query(self, text):
        # Let SentenceTransformer handle truncation internally
        return self.model.encode([text], truncate_dim=None)[0].tolist()


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
    ):
        """Initialize PDF RAG system.

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

        # Initialize embeddings with GPU support
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device} for embeddings")

        # Choose embedding model and determine model name for splitter
        if use_custom_embedding and Path(custom_embedding_path).exists():
            logger.info(f"Using custom trained embedding model: {custom_embedding_path}")
            # For custom models, we'll use sentence-transformers directly
            from sentence_transformers import SentenceTransformer
            self.custom_model = SentenceTransformer(custom_embedding_path)
            embedding_model_for_splitter = custom_embedding_path
            
            # Create a wrapper to make it compatible with LangChain
            self.embeddings = CustomEmbeddingWrapper(self.custom_model)
            logger.info("Custom embedding model loaded successfully")
        else:
            if use_custom_embedding:
                logger.warning(f"Custom embedding model not found at {custom_embedding_path}, using default")
            
            # Use default HuggingFace embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={"device": device},
                encode_kwargs={
                    "device": device,
                    "batch_size": 32,  # Smaller batch for longer sequences
                    "truncate_dim": None,  # Don't truncate
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

        # Create prompt template using ChatML format
        self.system_block = "<|im_start|>system\nYou are a helpful assistant specialised in master‑level STEM.\n<|im_end|>\n"
        self.user_template = """<|im_start|>user
The following is a question about knowledge and skills in advanced master‑level STEM courses.

Question: {question}

Context information is below:
{context}
<|im_end|>"""
        self.assistant_start = "<|im_start|>assistant\n <think>\n Okay, lets think."
        self.assistant_end = "\n<|im_end|>"

        logger.info("Model and components loaded")
        logger.info(f"RAG system ready with directory: {self.rag_dir}")
        logger.info(f"Using SentenceTransformersTokenTextSplitter with {self.embedding_chunk_size} tokens per chunk")

    def _load_vector_store(self):
        """Load existing vector store if available."""
        index_path = self.embeddings_dir / "faiss_index"
        logger.info(f"Checking for vector store at: {index_path}")

        if index_path.exists():
            try:
                logger.info("Loading existing vector store...")
                self.vector_store = FAISS.load_local(
                    str(index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True,  # Add this flag for newer versions
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
        """Save vector store to disk."""
        if self.vector_store is not None:
            try:
                index_path = self.embeddings_dir / "faiss_index"
                logger.info("Saving vector store...")
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

    def _chunk_text_data(
        self, 
        text_data: List[Any], 
        splitter: SentenceTransformersTokenTextSplitter, 
        is_training: bool = False
    ) -> List[Any]:
        """Chunk text data using the provided splitter.
        
        Responsibility: Pure chunking logic separated from extraction.
        This function is always executed in the main process to avoid CUDA issues.
        
        Args:
            text_data: List of text data to chunk
                - If is_training=False (vector store): [{"text": "...", "metadata": {...}}, ...]
                - If is_training=True: ["page_text_1", "page_text_2", ...]
            splitter: The SentenceTransformersTokenTextSplitter to use
            is_training: Whether this is for training (affects input/output format)
            
        Returns:
            Chunked data in the same format as input:
                - If is_training=False: [{"text": "chunk1", "metadata": {...}}, ...]
                - If is_training=True: ["chunk1", "chunk2", ...]
        """
        logger.debug(f"Chunking {len(text_data)} items with splitter (training={is_training})")
        
        chunks = []
        
        if is_training:
            # Training mode: input is ["text1", "text2", ...], output is ["chunk1", "chunk2", ...]
            for page_text in text_data:
                try:
                    text_chunks = splitter.split_text(page_text)
                    for chunk_text in text_chunks:
                        if chunk_text.strip() and len(chunk_text.strip()) >= 50:  # Filter short chunks for training
                            chunks.append(chunk_text.strip())
                except Exception as e:
                    logger.warning(f"Error chunking page text: {e}")
                    continue
        else:
            # Vector store mode: input is [{"text": "...", "metadata": {...}}, ...]
            for doc_data in text_data:
                try:
                    text_chunks = splitter.split_text(doc_data["text"])
                    for chunk_text in text_chunks:
                        if chunk_text.strip():  # Only add non-empty chunks
                            chunks.append({
                                "text": chunk_text.strip(), 
                                "metadata": doc_data["metadata"]
                            })
                except Exception as e:
                    logger.warning(f"Error chunking document: {e}")
                    continue
        
        logger.debug(f"Created {len(chunks)} chunks from {len(text_data)} input items")
        return chunks

    def add_pdf(self, pdf_path: str):
        """Add a PDF book to the vector store.

        Args:
            pdf_path: Path to the PDF file
        """
        # Check if PDF was already processed
        pdf_name = Path(pdf_path).name
        if pdf_name in self.processed_pdfs:
            logger.info(f"Skipping already processed PDF: {pdf_name}")
            return

        logger.info(f"Processing PDF: {pdf_path}")

        # Step 1: Extract text from PDF using modular approach
        page_data = extract_text_from_single_pdf(Path(pdf_path), include_metadata=True)

        if not page_data:
            logger.warning(f"No valid pages found in {pdf_name}")
            return

        # Step 2: Chunk text data using the modular chunking function
        chunks_with_metadata = self._chunk_text_data(
            page_data, self.text_splitter_for_embedding, is_training=False
        )

        if not chunks_with_metadata:
            logger.warning(f"No valid chunks found in {pdf_name}")
            return

        logger.info(f"Processing {len(chunks_with_metadata)} chunks with SentenceTransformer tokenization...")

        try:
            # Extract texts and metadatas for batch processing
            texts = [chunk["text"] for chunk in chunks_with_metadata]
            metadatas = [chunk["metadata"] for chunk in chunks_with_metadata]

            # Create or update vector store with batch processing
            if self.vector_store is None:
                logger.info("Creating new vector store...")
                self.vector_store = FAISS.from_texts(
                    texts, self.embeddings, metadatas=metadatas
                )
            else:
                logger.info("Adding to existing vector store...")
                self.vector_store.add_texts(texts, metadatas=metadatas)

            logger.info(f"Added {len(page_data)} pages ({len(chunks_with_metadata)} chunks) from {Path(pdf_path).stem}")

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
        """Retrieve relevant documents with scores and metadata.

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
            # Use similarity search with scores
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

            # Format results
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
        """Generate an answer based on retrieved documents using ChatML format."""
        # Retrieve relevant documents
        retrieved = self.retrieve_documents(
            question, k=k, score_threshold=score_threshold, filter_metadata=filter_metadata
        )

        if not retrieved:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "retrieval_scores": [],
                "context_docs": 0,
            }

        # Format context from retrieved documents
        context = "\n\n".join(
            f"From {doc['book']} (Page {doc['page']}):\n{doc['content']}"
            for doc in retrieved
        )

        # Create prompt using the ChatML format defined in __init__
        user_prompt = self.user_template.format(question=question, context=context)
        full_prompt = self.system_block + user_prompt + self.assistant_start

        # Generate answer using invoke
        response = self.llm.invoke(full_prompt)
        
        # Clean up the response to remove any thinking tokens or extra formatting
        answer = response.strip()
        if answer.startswith("<think>"):
            # Find the end of thinking and extract the actual answer
            think_end = answer.find("</think>")
            if think_end != -1:
                answer = answer[think_end + 8:].strip()
        
        # Remove any trailing end tokens
        if answer.endswith(self.assistant_end.strip()):
            answer = answer[:-len(self.assistant_end.strip())].strip()

        return {
            "answer": answer,
            "sources": [doc["book"] for doc in retrieved],
            "retrieval_scores": [doc["score"] for doc in retrieved],
            "context_docs": len(retrieved),
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
        """Add multiple PDFs efficiently with automatic processing mode selection.
        
        This method provides a unified interface for processing multiple PDFs:
        - Parallel processing (default): Fast extraction using multiple workers
        - Sequential processing (fallback): Reliable single-threaded processing
        - CUDA-safe: All chunking happens in main process
        - Efficient batching for vector store operations
        
        Args:
            pdf_paths: List of PDF file paths
            max_workers: Number of parallel workers (defaults to CPU count)
            batch_size: Number of chunks to process in each batch
            use_parallel: Whether to use parallel processing (default: True)
        """
        logger.info(f"Processing {len(pdf_paths)} PDFs (parallel={use_parallel}, batch_size={batch_size})")

        # Filter out already processed PDFs
        unprocessed_pdfs = self._filter_unprocessed_pdfs(pdf_paths)

        if not unprocessed_pdfs:
            logger.info("All PDFs already processed!")
            return

        logger.info(f"Processing {len(unprocessed_pdfs)} new PDFs...")
        start_time = time.time()

        # Step 1: Extract text from all PDFs
        if use_parallel:
            logger.info(f"🚀 Using parallel extraction with {max_workers or 'auto'} workers")
            try:
                pdf_paths_as_path_objects = [Path(p) for p in unprocessed_pdfs]
                all_page_data = extract_text_from_multiple_pdfs(
                    pdf_paths_as_path_objects, include_metadata=True, max_workers=max_workers
                )
            except Exception as e:
                logger.warning(f"Parallel processing failed: {e}. Falling back to sequential...")
                use_parallel = False
        
        if not use_parallel:
            logger.info("📝 Using sequential extraction")
            all_page_data = []
            for pdf_path in unprocessed_pdfs:
                try:
                    page_data = extract_text_from_single_pdf(Path(pdf_path), include_metadata=True)
                    all_page_data.extend(page_data)
                    logger.info(f"Extracted {len(page_data)} pages from {Path(pdf_path).name}")
                except Exception as e:
                    logger.error(f"Error extracting from {Path(pdf_path).name}: {e}")
                    continue

        if not all_page_data:
            logger.warning("No valid page data extracted!")
            return

        logger.info(f"✅ Extracted {len(all_page_data)} pages total")

        # Step 2: Chunk all text in main process
        logger.info(f"📝 Chunking {len(all_page_data)} pages...")
        chunks_with_metadata = self._chunk_text_data(
            all_page_data, self.text_splitter_for_embedding, is_training=False
        )

        if not chunks_with_metadata:
            logger.warning("No valid chunks created!")
            return

        logger.info(f"Created {len(chunks_with_metadata)} chunks")

        # Step 3: Add chunks to vector store in batches
        logger.info(f"💾 Adding {len(chunks_with_metadata)} chunks to vector store (batch_size={batch_size})")

        try:
            all_chunks = [chunk["text"] for chunk in chunks_with_metadata]
            all_metadatas = [chunk["metadata"] for chunk in chunks_with_metadata]
            
            for i in range(0, len(all_chunks), batch_size):
                batch_chunks = all_chunks[i:i + batch_size]
                batch_metadatas = all_metadatas[i:i + batch_size]
                
                if self.vector_store is None:
                    logger.info("Creating new vector store...")
                    self.vector_store = FAISS.from_texts(
                        batch_chunks, self.embeddings, metadatas=batch_metadatas
                    )
                else:
                    batch_num = i//batch_size + 1
                    total_batches = (len(all_chunks) + batch_size - 1)//batch_size
                    logger.info(f"Adding batch {batch_num}/{total_batches}...")
                    self.vector_store.add_texts(batch_chunks, metadatas=batch_metadatas)

            # Mark all PDFs as processed
            processed_files = [Path(pdf_path).name for pdf_path in unprocessed_pdfs]
            for pdf_name in processed_files:
                self.processed_pdfs.add(pdf_name)

            self._save_processed_pdfs()
            self._save_vector_store()

            total_time = time.time() - start_time
            mode_str = "parallel" if use_parallel else "sequential"
            logger.info(f"🎉 Successfully processed {len(processed_files)} PDFs with {len(all_chunks)} chunks")
            logger.info(f"⏱️  Total time: {total_time:.2f}s | Speed: {len(all_chunks)/total_time:.2f} chunks/s ({mode_str})")

        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}")
            raise

    def train_custom_embedding_model(
        self, 
        base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        push_to_hub: bool = False,
        hub_model_id: Optional[str] = None,
        hub_token: Optional[str] = None,
        hub_private: bool = False
    ) -> bool:
        """Train a custom embedding model using the PDF corpus with SentenceTransformers v3+ API.
        
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
            
            logger.info("Training custom embedding model with SentenceTransformers v3+ API...")
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
            
            # 2. Prepare training corpus from PDFs
            logger.info("Preparing training corpus from PDFs...")
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
            
            # Alternative: if you specifically need TSDAE, you might need to use the old fit method
            # For now, let's use MultipleNegativesRankingLoss which is more compatible with v3+ API
            
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
                "training_api": "SentenceTransformers_v3+",
                "loss_function": "MultipleNegativesRankingLoss",
                "eval_strategy": "steps",  # Updated parameter name
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
            logger.info("Custom embedding model trained successfully!")
            
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

    def _prepare_training_corpus_from_pdfs(self) -> List[str]:
        """Prepare training corpus from PDF files using modular extraction and chunking."""
        pdf_paths = list(self.rag_dir.glob("*.pdf"))
        if not pdf_paths:
            logger.error(f"No PDF files found in {self.rag_dir}")
            return []

        logger.info(f"Found {len(pdf_paths)} PDF files to process for training")

        # Step 1: Extract text from all PDFs (parallel, no metadata for training)
        all_page_texts = extract_text_from_multiple_pdfs(pdf_paths, include_metadata=False, max_workers=None)

        if not all_page_texts:
            logger.warning("No text extracted from PDFs for training")
            return []

        logger.info(f"Extracted {len(all_page_texts)} pages for training")

        # Step 2: Chunk all text using the modular chunking function
        training_chunks = self._chunk_text_data(
            all_page_texts, self.text_splitter_for_training, is_training=True
        )

        logger.info(f"Prepared {len(training_chunks)} training chunks")
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