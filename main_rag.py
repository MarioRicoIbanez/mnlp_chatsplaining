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
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

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

print(f"Script location: {SCRIPT_DIR}")
print(f"Project root: {PROJECT_ROOT}")
print(f"RAG directory: {RAG_DIR}")
print(f"Embeddings directory: {EMBEDDINGS_DIR}")

# Load environment variables from .env file (look in script directory first)
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    load_dotenv(env_file)
    print(f"Loaded .env from: {env_file}")
else:
    load_dotenv()  # Try default locations
    print("Loaded .env from default locations")

 
def find_rag_directory() -> Path:
    """Find the RAG directory containing PDF files."""
    
    # Possible locations to search for RAG directory
    search_locations = [
        # Current directory structure
        RAG_DIR,
        # Results model directory
        RESULTS_DIR / "RAG", 
        # Parent directories
        PROJECT_ROOT.parent / "RAG",
        PROJECT_ROOT.parent / "results_model" / "RAG",
    ]
    
    # Also search recursively from project root
    for subdir in PROJECT_ROOT.rglob("RAG"):
        if subdir.is_dir():
            search_locations.append(subdir)
    
    # Check each location
    for location in search_locations:
        if location.exists() and location.is_dir():
            pdf_files = list(location.glob("*.pdf"))
            if len(pdf_files) > 0:
                logger.info(f"Found RAG directory with {len(pdf_files)} PDFs at: {location}")
                return location
    
    # Default: use the project RAG directory (create if needed)
    logger.warning(f"No existing RAG directory with PDFs found, using default: {RAG_DIR}")
    RAG_DIR.mkdir(exist_ok=True)
    return RAG_DIR


def extract_and_chunk_pdf(args):
    """Worker function for multiprocessing PDF extraction and chunking."""
    pdf_path, max_chunk_size, chunk_overlap = args

    try:
        # Get book title from filename
        book_title = Path(pdf_path).stem

        # Extract text from PDF
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)

            documents = []
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text = page.extract_text()

                if text.strip():  # Only add non-empty pages
                    documents.append(
                        {
                            "text": text,
                            "metadata": {
                                "book": book_title,
                                "page": page_num + 1,
                                "total_pages": num_pages,
                            },
                        }
                    )

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )

        chunks = []
        for doc in documents:
            text_chunks = text_splitter.split_text(doc["text"])

            for chunk in text_chunks:
                if chunk.strip():  # Only add non-empty chunks
                    chunks.append({"text": chunk, "metadata": doc["metadata"]})

        return pdf_path, len(documents), chunks

    except Exception as e:
        return pdf_path, 0, f"Error: {str(e)}"


class PDFRAG:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_chunk_size: int = 1024,
        chunk_overlap: int = 100,
        similarity_threshold: float = 0.7,
        embeddings_dir: Optional[str] = None,
        use_custom_embedding: bool = False,
        custom_embedding_path: Optional[str] = None,
    ):
        """Initialize PDF RAG system.

        Args:
            model_name: HuggingFace model name for the LLM
            embedding_model: HuggingFace model name for embeddings
            max_chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Overlap between chunks
            similarity_threshold: Default threshold for document similarity (0-1)
            embeddings_dir: Directory to save/load embeddings (defaults to project structure)
            use_custom_embedding: Whether to use custom trained embedding model
            custom_embedding_path: Path to custom trained embedding model (defaults to project structure)
        """
        logger.info(f"Loading model from HuggingFace: {model_name}")
        logger.info("Project structure:")
        logger.info(f"   Script: {SCRIPT_DIR}")
        logger.info(f"   Project root: {PROJECT_ROOT}")
        logger.info(f"   Results: {RESULTS_DIR}")

        # Auto-detect RAG directory if not provided
        detected_rag_dir = find_rag_directory()
        
        # Set up paths with auto-detection and project-relative defaults
        if embeddings_dir is None:
            embeddings_dir = str(EMBEDDINGS_DIR)
        if custom_embedding_path is None:
            custom_embedding_path = str(EMBEDDINGS_DIR / "custom_model")
            
        logger.info(f"Using embeddings directory: {embeddings_dir}")
        logger.info(f"Using custom embedding path: {custom_embedding_path}")

        # Set up embeddings directory
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        # Store detected RAG directory for later use
        self.rag_dir = detected_rag_dir
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

        # Initialize text splitter with token-based length function
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._count_tokens,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )

        # Initialize embeddings with GPU support
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device} for embeddings")
        
        # Store embedding configuration
        self.use_custom_embedding = use_custom_embedding
        self.custom_embedding_path = custom_embedding_path

        # Choose embedding model
        if use_custom_embedding and Path(custom_embedding_path).exists():
            logger.info(f"Using custom trained embedding model: {custom_embedding_path}")
            # For custom models, we'll use sentence-transformers directly
            from sentence_transformers import SentenceTransformer
            self.custom_model = SentenceTransformer(custom_embedding_path)
            
            # Create a wrapper to make it compatible with LangChain
            class CustomEmbeddingWrapper:
                def __init__(self, model):
                    self.model = model
                    self.max_seq_length = getattr(model, 'max_seq_length', 512)
                    
                def _truncate_text(self, text: str) -> str:
                    """Truncate text to fit model's max sequence length."""
                    tokens = self.model.tokenizer.encode(text, add_special_tokens=False)
                    if len(tokens) > self.max_seq_length - 2:  # Account for special tokens
                        tokens = tokens[:self.max_seq_length - 2]
                        text = self.model.tokenizer.decode(tokens, skip_special_tokens=True)
                    return text
                    
                def embed_documents(self, texts):
                    # Truncate texts that are too long
                    truncated_texts = [self._truncate_text(text) for text in texts]
                    return self.model.encode(truncated_texts).tolist()
                    
                def embed_query(self, text):
                    truncated_text = self._truncate_text(text)
                    return self.model.encode([truncated_text])[0].tolist()
            
            self.embeddings = CustomEmbeddingWrapper(self.custom_model)
            logger.info("Custom embedding model loaded successfully")
        else:
            if use_custom_embedding:
                logger.warning(f"Custom embedding model not found at {custom_embedding_path}, using default")
            
            # Use default HuggingFace embeddings with longer sequence support
            if embedding_model == "sentence-transformers/all-MiniLM-L6-v2":
                # Switch to a model that supports longer sequences
                logger.warning("Switching to all-mpnet-base-v2 for better 1024-token support")
                embedding_model = "sentence-transformers/all-mpnet-base-v2"
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={"device": device},
                encode_kwargs={
                    "device": device,
                    "batch_size": 32,  # Smaller batch for longer sequences
                    "truncate_dim": None,  # Don't truncate
                },
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

        # Initialize tokenizer for length counting
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

        # Initialize LLM
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
        self.assistant_start = "<|im_start|>assistant\n"
        self.assistant_end = "\n<|im_end|>"

        logger.info("Model and components loaded")
        logger.info(f"RAG system ready with directory: {self.rag_dir}")

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text."""
        return len(self.tokenizer.encode(text))
    
    def _truncate_to_embedding_limit(self, text: str, max_tokens: int = 512) -> str:
        """Truncate text to fit embedding model limits."""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate to max_tokens and decode back
        truncated_tokens = tokens[:max_tokens]
        return self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)

    def _extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF with metadata.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of dictionaries containing text chunks and metadata
        """
        logger.info(f"Processing PDF: {pdf_path}")

        # Get book title from filename
        book_title = Path(pdf_path).stem

        # Open PDF
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)

            # Extract text from each page
            documents = []
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text = page.extract_text()

                if text.strip():  # Only add non-empty pages
                    documents.append(
                        {
                            "text": text,
                            "metadata": {
                                "book": book_title,
                                "page": page_num + 1,
                                "total_pages": num_pages,
                            },
                        }
                    )

        return documents

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
            logger.info("INFO: No existing vector store found")

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

        # Extract text from PDF
        documents = self._extract_text_from_pdf(pdf_path)

        # Split documents into chunks
        chunks = []
        for doc in documents:
            # Split text into chunks
            text_chunks = self.text_splitter.split_text(doc["text"])

            # Create chunks with metadata, ensuring they fit embedding limits
            for chunk in text_chunks:
                if chunk.strip():  # Only add non-empty chunks
                    # Truncate chunk to fit embedding model limits
                    truncated_chunk = self._truncate_to_embedding_limit(chunk.strip(), max_tokens=512)
                    chunks.append({"text": truncated_chunk, "metadata": doc["metadata"]})

        if not chunks:
            logger.warning(f"No valid chunks found in {pdf_name}")
            return

        logger.info(f"Processing {len(chunks)} chunks with GPU acceleration...")

        try:
            # Extract texts and metadatas for batch processing
            texts = [chunk["text"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]

            # Create or update vector store with batch processing
            if self.vector_store is None:
                logger.info("Creating new vector store...")
                self.vector_store = FAISS.from_texts(
                    texts, self.embeddings, metadatas=metadatas
                )
            else:
                logger.info("Adding to existing vector store...")
                self.vector_store.add_texts(texts, metadatas=metadatas)

            logger.info(
                f"Added {len(documents)} pages ({len(chunks)} chunks) from {Path(pdf_path).stem}"
            )

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
        k: int = 3,
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
            logger.error("ERROR: No vector store available for retrieval")
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
            logger.error(f"ERROR: Error during document retrieval: {e}")
            import traceback

            traceback.print_exc()
            return []

    def generate_answer(
        self,
        question: str,
        k: int = 3,
        score_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate answer using retrieved context.

        Args:
            question: The question to answer
            k: Number of documents to retrieve
            score_threshold: Optional minimum similarity score (0-1)
            filter_metadata: Optional metadata filters to apply

        Returns:
            Dictionary containing answer, sources, and retrieval information
        """
        if self.vector_store is None:
            return {
                "answer": "No books have been added to the system yet.",
                "context_docs": 0,
                "sources": [],
                "retrieval_scores": [],
            }

        # Get retrieved documents
        retrieved_docs = self.retrieve_documents(
            question,
            k=k,
            score_threshold=score_threshold or self.similarity_threshold,
            filter_metadata=filter_metadata,
        )

        if not retrieved_docs:
            return {
                "answer": "No relevant information found in the books.",
                "context_docs": 0,
                "sources": [],
                "retrieval_scores": [],
            }

        # Format context from retrieved documents
        context = "\n\n".join(
            [
                f"From {doc['book']} (Page {doc['page']}):\n{doc['content']}"
                for doc in retrieved_docs
            ]
        )

        # Create prompt using ChatML format
        prompt = (
            self.system_block
            + self.user_template.format(question=question, context=context)
            + self.assistant_start
        )

        # Generate answer using the LLM
        response = self.llm(prompt)

        # Extract sources and scores
        sources = []
        scores = []
        for doc in retrieved_docs:
            source = f"{doc['book']} (Page {doc['page']})"
            sources.append(source)
            scores.append(doc["score"])

        return {
            "answer": response,
            "context_docs": len(retrieved_docs),
            "sources": sources,
            "retrieval_scores": scores,
        }

    def debug_vector_store(self):
        """Debug function to analyze vector store contents."""
        if self.vector_store is None:
            logger.error("ERROR: No vector store to debug")
            return

        logger.info("INFO: Vector Store Debug Information:")
        logger.info(f"Total vectors: {self.vector_store.index.ntotal}")

        # Get a sample of documents to see what's in the store
        try:
            # Get all document metadata
            docstore = self.vector_store.docstore
            logger.info(f"INFO: Docstore size: {len(docstore._dict)}")

            # Count documents by book
            book_counts = {}
            for doc_id, doc in docstore._dict.items():
                book_name = doc.metadata.get("book", "Unknown")
                book_counts[book_name] = book_counts.get(book_name, 0) + 1

            logger.info("INFO: Documents by book:")
            for book, count in sorted(book_counts.items()):
                logger.info(f"  {book}: {count} chunks")

            # Test embedding quality with a simple query
            logger.info("INFO: Testing embedding quality:")
            test_queries = [
                "machine learning",
                "neural networks",
                "backpropagation",
                "gradient descent",
            ]

            for query in test_queries:
                docs = self.vector_store.similarity_search_with_score(query, k=3)
                logger.info(f"\nQuery: '{query}'")
                for i, (doc, score) in enumerate(docs):
                    book = doc.metadata.get("book", "Unknown")
                    page = doc.metadata.get("page", "Unknown")
                    content_preview = doc.page_content[:100].replace("\n", " ")
                    logger.info(f"  {i+1}. {book} (Page {page}) - Score: {score:.3f}")
                    logger.info(f"     Content: {content_preview}...")

        except Exception as e:
            logger.error("ERROR: Error during debug: {e}")
            import traceback

            traceback.print_exc()

    def rebuild_vector_store(self):
        """Rebuild the vector store from scratch."""
        logger.info("INFO: Rebuilding vector store from scratch...")

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
            logger.info("INFO: Cleared existing vector store")
        except Exception as e:
            logger.warning(f"WARNING: Error clearing existing data: {e}")

    def add_multiple_pdfs(self, pdf_paths: List[str], max_workers: int = 2):
        """Add multiple PDFs efficiently.

        Args:
            pdf_paths: List of PDF file paths
            max_workers: Number of parallel workers for PDF text extraction
        """
        logger.info(f"INFO: Processing {len(pdf_paths)} PDFs with {max_workers} workers...")

        # Filter out already processed PDFs
        unprocessed_pdfs = []
        for pdf_path in pdf_paths:
            pdf_name = Path(pdf_path).name
            if pdf_name not in self.processed_pdfs:
                unprocessed_pdfs.append(pdf_path)
            else:
                logger.info(f"INFO: Skipping already processed: {pdf_name}")

        if not unprocessed_pdfs:
            logger.info("INFO: All PDFs already processed!")
            return

        logger.info(f"INFO: Processing {len(unprocessed_pdfs)} new PDFs...")

        # Process each PDF (keep sequential for now to avoid memory issues)
        for pdf_path in unprocessed_pdfs:
            try:
                self.add_pdf(pdf_path)
            except Exception as e:
                logger.error(f"ERROR: Failed to process {Path(pdf_path).name}: {str(e)}")
                continue

    def add_multiple_pdfs_parallel(self, pdf_paths: List[str], max_workers: int = None):
        """Add multiple PDFs using parallel processing for much faster performance.

        Args:
            pdf_paths: List of PDF file paths
            max_workers: Number of parallel workers (defaults to CPU count)
        """
        if max_workers is None:
            max_workers = min(mp.cpu_count(), len(pdf_paths))

        logger.info(
            f"INFO: Processing {len(pdf_paths)} PDFs with {max_workers} parallel workers..."
        )

        # Filter out already processed PDFs
        unprocessed_pdfs = []
        for pdf_path in pdf_paths:
            pdf_name = Path(pdf_path).name
            if pdf_name not in self.processed_pdfs:
                unprocessed_pdfs.append(pdf_path)
            else:
                logger.info(f"INFO: Skipping already processed: {pdf_name}")

        if not unprocessed_pdfs:
            logger.info("INFO: All PDFs already processed!")
            return

        logger.info(f"INFO: Processing {len(unprocessed_pdfs)} new PDFs in parallel...")

        # Prepare arguments for multiprocessing
        args = [(pdf_path, 1024, 100) for pdf_path in unprocessed_pdfs]

        # Process PDFs in parallel
        all_chunks = []
        all_metadatas = []
        processed_files = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_path = {
                executor.submit(extract_and_chunk_pdf, arg): arg[0] for arg in args
            }

            # Collect results as they complete
            for future in as_completed(future_to_path):
                pdf_path = future_to_path[future]
                pdf_name = Path(pdf_path).name

                try:
                    pdf_path_result, num_pages, chunks = future.result()

                    if isinstance(chunks, str):  # Error case
                        logger.error(f"ERROR: {pdf_name}: {chunks}")
                        continue

                    logger.info(f"INFO: {pdf_name}: {num_pages} pages, {len(chunks)} chunks")

                    # Collect chunks and metadata
                    for chunk in chunks:
                        all_chunks.append(chunk["text"])
                        all_metadatas.append(chunk["metadata"])

                    processed_files.append(pdf_name)

                except Exception as e:
                    logger.error(f"ERROR: {pdf_name}: {str(e)}")

        if not all_chunks:
            logger.warning("WARNING: No valid chunks to add!")
            return

        # Add all chunks to vector store in one go (much faster)
        logger.info(
            f"INFO: Adding {len(all_chunks)} chunks to vector store with GPU acceleration..."
        )

        try:
            if self.vector_store is None:
                logger.info("INFO: Creating new vector store...")
                self.vector_store = FAISS.from_texts(
                    all_chunks, self.embeddings, metadatas=all_metadatas
                )
            else:
                logger.info("INFO: Adding to existing vector store...")
                self.vector_store.add_texts(all_chunks, metadatas=all_metadatas)

            # Mark all PDFs as processed
            for pdf_name in processed_files:
                self.processed_pdfs.add(pdf_name)

            self._save_processed_pdfs()
            self._save_vector_store()

            logger.info(
                f"INFO: Successfully processed {len(processed_files)} PDFs with {len(all_chunks)} total chunks"
            )

        except Exception as e:
            logger.error(f"ERROR: Error adding chunks to vector store: {e}")
            raise
    
    def train_custom_embedding_model(self, base_model: str = "sentence-transformers/all-mpnet-base-v2") -> bool:
        """Train a custom embedding model using the PDF corpus."""
        try:
            from utils.embedding_utils import train_custom_embedding_model
            
            # Use project-relative paths
            pdf_dir = str(self.rag_dir)
            output_dir = str(self.embeddings_dir / "custom_model")
            
            logger.info("INFO: Training custom embedding model...")
            logger.info(f"INFO: PDF directory: {pdf_dir}")
            logger.info(f"INFO: Output directory: {output_dir}")
            logger.info(f"INFO: Base model: {base_model}")
            
            success = train_custom_embedding_model(
                pdf_dir=pdf_dir,
                output_dir=output_dir,
                base_model=base_model
            )
            
            if success:
                logger.info("INFO: Custom embedding model trained successfully!")
                logger.info("INFO: To use it, initialize PDFRAG with:")
                logger.info(f"   use_custom_embedding=True")
                logger.info(f"   custom_embedding_path='{output_dir}'")
                return True
            else:
                logger.error("ERROR: Custom embedding model training failed!")
                return False
                
        except ImportError as e:
            logger.error(f"ERROR: Could not import embedding utilities: {e}")
            logger.info("INFO: Install required packages: pip install sentence-transformers scikit-learn")
            return False
        except Exception as e:
            logger.error(f"ERROR: Error training custom embedding model: {e}")
            return False


def main():
    """Example usage of the PDF RAG system."""
    import sys
    import time

    logger.info("PDF RAG System Starting...")
    logger.info("Project structure:")
    logger.info(f"   Script: {SCRIPT_DIR}")
    logger.info(f"   Project root: {PROJECT_ROOT}")
    logger.info(f"   RAG directory: {RAG_DIR}")
    logger.info(f"   Results: {RESULTS_DIR}")
    logger.info(f"   Embeddings: {EMBEDDINGS_DIR}")

    # Check command line arguments
    rebuild = "--rebuild" in sys.argv
    train_embedding = "--train-embedding" in sys.argv

    # Initialize RAG system with optimized settings
    rag = PDFRAG(max_chunk_size=1024, similarity_threshold=0.1)

    if rebuild:
        logger.info("Fast Embedding Rebuild Mode")
        logger.info("=" * 50)

        # Force rebuild
        logger.info("Clearing existing embeddings...")
        rag.rebuild_vector_store()

        # Get all PDFs using the stored RAG directory
        pdf_files = list(rag.rag_dir.glob("*.pdf"))

        logger.info(f"Found {len(pdf_files)} PDF files in {rag.rag_dir}")
        for i, pdf_file in enumerate(pdf_files, 1):
            size_mb = pdf_file.stat().st_size / (1024 * 1024)
            logger.info(f"  {i}. {pdf_file.name} ({size_mb:.1f} MB)")

        if not pdf_files:
            logger.warning(f"No PDF files found in {rag.rag_dir}")
            logger.info("Add PDF files to the RAG directory and try again")
            return

        # Process all PDFs in parallel
        logger.info("Starting parallel processing with GPU acceleration...")
        start_time = time.time()

        # Use parallel processing for much faster performance
        rag.add_multiple_pdfs_parallel([str(f) for f in pdf_files])

        # Final summary
        total_time = time.time() - start_time
        if rag.vector_store:
            total_vectors = rag.vector_store.index.ntotal
            logger.info("Rebuild complete!")
            logger.info(f"Total time: {total_time:.1f} seconds")
            logger.info(f"Total vectors: {total_vectors}")
            logger.info(f"Speed: {total_vectors/total_time:.1f} vectors/second")

            # Quick test
            logger.info("Quick test retrieval:")
            test_results = rag.retrieve_documents("machine learning", k=3)
            for doc in test_results:
                logger.info(f"  {doc['book']} (Page {doc['page']}) - Score: {doc['score']:.3f}")
        else:
            logger.error(f"Rebuild failed after {total_time:.1f} seconds")

        return  # Exit after rebuild

    if train_embedding:
        logger.info("Custom Embedding Training Mode")
        logger.info("=" * 50)
        
        # Check if PDFs exist first
        pdf_files = list(rag.rag_dir.glob("*.pdf"))
        if not pdf_files:
            logger.error(f"No PDF files found in {rag.rag_dir}")
            logger.info("Add PDF files to the RAG directory first")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files for training")
        
        # Train custom embedding model
        success = rag.train_custom_embedding_model()
        
        if success:
            logger.info("Embedding training completed!")
            logger.info("Now run with --rebuild to use the custom model:")
            logger.info("   python main_rag.py --rebuild")
        else:
            logger.error("Embedding training failed!")
            
        return  # Exit after training

    # Normal processing mode
    # Add PDF books efficiently using the stored RAG directory
    pdf_files = list(rag.rag_dir.glob("*.pdf"))

    logger.info(f"Found {len(pdf_files)} PDF files in {rag.rag_dir}")

    if not pdf_files:
        logger.warning(f"No PDF files found in {rag.rag_dir}")
        logger.info("Add PDF files to the RAG directory:")
        logger.info(f"   cp your_pdfs/*.pdf {rag.rag_dir}/")
        return

    # Use the optimized batch processing
    rag.add_multiple_pdfs([str(f) for f in pdf_files])

    # Debug: Check vector store status
    if rag.vector_store is not None:
        logger.info(f"Vector store status: LOADED with {rag.vector_store.index.ntotal} vectors")
        rag.debug_vector_store()  # Add debug analysis
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
        logger.error("Vector store status: NOT LOADED")
        logger.error("Vector store not loaded - cannot run Q&A session")
        return
    
    # Example questions
    questions = [
        "What is the definition of machine learning?",
        "Explain the concept of backpropagation.",
        "What are the main types of neural networks?",
        "How does gradient descent work?",
        "What is the atommic commit?",
        "What is the spleen used for?",
        "What is the baby step giant step algorithm?",
    ]
    
    for question in questions:
        logger.info(f"\n{'='*80}")
        logger.info(f"Question: {question}")
        logger.info("-" * 80)

        # Show retrieved documents with no score threshold for debugging
        logger.info("Retrieved Documents (no threshold):")
        retrieved = rag.retrieve_documents(question, k=5, score_threshold=None)
        logger.info("=" * 80)
        logger.info(f"Question: {question}")
        logger.info("-" * 80)
        
        # Show retrieved documents
        logger.info("Retrieved Documents:")
        retrieved = rag_instance.retrieve_documents(question, k=10, score_threshold=0.5)
        for i, doc in enumerate(retrieved):
            logger.info(f"\n{i+1}. Book: {doc['book']}")
            logger.info(f"   Page: {doc['page']}")
            logger.info(f"   Score: {doc['score']:.3f}")
            logger.info(f"   Content: {doc['content'][:200]}...")

            logger.info(f"{i+1}. Book: {doc['book']} | Page: {doc['page']} | Score: {doc['score']:.3f}")
            logger.info(f"   Content: {doc['content'][:200]}...")
        
        if not retrieved:
            logger.warning("No documents retrieved! There might be an issue with the embeddings.")
            logger.warning("No documents retrieved! There might be an issue with the embeddings.")
            continue

        # Generate answer
        result = rag.generate_answer(question, score_threshold=None)

        logger.info(f"Answer:\n{result['answer']}")
        logger.info("Sources:")
        
        # Generate answer using LangChain prompt templates
        result = rag_instance.generate_answer(question, score_threshold=0.5)
        
        # Use print for final answer (user-facing output)
        print(f"\n🎯 Answer:\n{result['answer']}\n")
        
        # Use logger for sources (operational information)
        logger.info("Sources:")
        for source, score in zip(result["sources"], result["retrieval_scores"]):
            logger.info(f"- {source} (Score: {score:.3f})")
        logger.info(f"Documents used: {result['context_docs']}")
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