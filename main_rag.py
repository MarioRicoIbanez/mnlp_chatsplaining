#!/usr/bin/env python3
"""
PDF RAG system for processing and retrieving information from PDF books.
Features:
- PDF text extraction
- Smart chunking with max size of 512 tokens
- Metadata preservation (book title, chapter, page)
- Enhanced retrieval with book-specific context
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


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
        max_chunk_size: int = 512,
        chunk_overlap: int = 50,
        similarity_threshold: float = 0.7,
        embeddings_dir: str = "RAG/embeddings",
    ):
        """Initialize PDF RAG system.

        Args:
            model_name: HuggingFace model name for the LLM
            embedding_model: HuggingFace model name for embeddings
            max_chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Overlap between chunks
            similarity_threshold: Default threshold for document similarity (0-1)
            embeddings_dir: Directory to save/load embeddings
        """
        print(f"🤖 Loading model from HuggingFace: {model_name}")

        # Load environment variables
        load_dotenv()

        # Set up embeddings directory
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_dir.mkdir(exist_ok=True)

        # Initialize text splitter with token-based length function
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._count_tokens,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )

        # Initialize embeddings with GPU support and faster model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Using device: {device} for embeddings")

        # Use a smaller, faster model for embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",  # Much faster than sentence-transformers version
            model_kwargs={"device": device},
            encode_kwargs={
                "device": device,
                "batch_size": 64,
            },  # Large batch size for speed
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Initialize LLM
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side="right"
        )
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
            temperature=0.7,
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

        print("✅ Model and components loaded")

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text."""
        return len(self.tokenizer.encode(text))

    def _extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF with metadata.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of dictionaries containing text chunks and metadata
        """
        print(f"📖 Processing PDF: {pdf_path}")

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
        print(f"🔍 Checking for vector store at: {index_path}")

        if index_path.exists():
            try:
                print("📚 Loading existing vector store...")
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
                    print(
                        f"✅ Vector store loaded successfully with {self.vector_store.index.ntotal} vectors"
                    )
                else:
                    print("⚠️ Vector store loaded but appears to be empty")
                    self.vector_store = None

            except Exception as e:
                print(f"⚠️ Could not load vector store: {e}")
                print(f"⚠️ Error type: {type(e).__name__}")
                import traceback

                traceback.print_exc()
                self.vector_store = None
        else:
            print("📁 No existing vector store found")

    def _save_vector_store(self):
        """Save vector store to disk."""
        if self.vector_store is not None:
            try:
                index_path = self.embeddings_dir / "faiss_index"
                print("💾 Saving vector store...")
                self.vector_store.save_local(str(index_path))
                print(
                    f"✅ Vector store saved successfully with {self.vector_store.index.ntotal} vectors"
                )
            except Exception as e:
                print(f"⚠️ Could not save vector store: {e}")

    def _load_processed_pdfs(self):
        """Load list of processed PDFs."""
        processed_file = self.embeddings_dir / "processed_pdfs.txt"
        if processed_file.exists():
            with open(processed_file, "r") as f:
                self.processed_pdfs = set(line.strip() for line in f)
            print(f"📚 Loaded {len(self.processed_pdfs)} processed PDFs")

    def _save_processed_pdfs(self):
        """Save list of processed PDFs."""
        processed_file = self.embeddings_dir / "processed_pdfs.txt"
        with open(processed_file, "w") as f:
            for pdf in sorted(self.processed_pdfs):
                f.write(f"{pdf}\n")
        print(f"💾 Saved {len(self.processed_pdfs)} processed PDFs")

    def add_pdf(self, pdf_path: str):
        """Add a PDF book to the vector store.

        Args:
            pdf_path: Path to the PDF file
        """
        # Check if PDF was already processed
        pdf_name = Path(pdf_path).name
        if pdf_name in self.processed_pdfs:
            print(f"📚 Skipping already processed PDF: {pdf_name}")
            return

        print(f"📖 Processing PDF: {pdf_path}")

        # Extract text from PDF
        documents = self._extract_text_from_pdf(pdf_path)

        # Split documents into chunks
        chunks = []
        for doc in documents:
            # Split text into chunks
            text_chunks = self.text_splitter.split_text(doc["text"])

            # Create chunks with metadata
            for chunk in text_chunks:
                if chunk.strip():  # Only add non-empty chunks
                    chunks.append({"text": chunk, "metadata": doc["metadata"]})

        if not chunks:
            print(f"⚠️ No valid chunks found in {pdf_name}")
            return

        print(f"🔄 Processing {len(chunks)} chunks with GPU acceleration...")

        try:
            # Extract texts and metadatas for batch processing
            texts = [chunk["text"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]

            # Create or update vector store with batch processing
            if self.vector_store is None:
                print("🆕 Creating new vector store...")
                self.vector_store = FAISS.from_texts(
                    texts, self.embeddings, metadatas=metadatas
                )
            else:
                print("➕ Adding to existing vector store...")
                self.vector_store.add_texts(texts, metadatas=metadatas)

            print(
                f"✅ Added {len(documents)} pages ({len(chunks)} chunks) from {Path(pdf_path).stem}"
            )

            # Mark PDF as processed and save
            self.processed_pdfs.add(pdf_name)
            self._save_processed_pdfs()

            # Save updated vector store
            self._save_vector_store()

        except Exception as e:
            print(f"❌ Error processing {pdf_name}: {str(e)}")
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
        print(
            f"🔍 Retrieving documents for query: '{query[:50]}...' (vector_store exists: {self.vector_store is not None})"
        )

        if self.vector_store is None:
            print("❌ No vector store available for retrieval")
            return []

        try:
            # Use similarity search with scores
            docs_and_scores = self.vector_store.similarity_search_with_score(
                query, k=k, filter=filter_metadata
            )

            print(f"📚 Found {len(docs_and_scores)} documents before filtering")

            # Filter by score threshold if specified
            if score_threshold is not None:
                docs_and_scores = [
                    (doc, score)
                    for doc, score in docs_and_scores
                    if score >= score_threshold
                ]
                print(
                    f"📚 {len(docs_and_scores)} documents after score filtering (threshold: {score_threshold})"
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
            print(f"❌ Error during document retrieval: {e}")
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
            print("❌ No vector store to debug")
            return

        print(f"\n🔍 Vector Store Debug Information:")
        print(f"Total vectors: {self.vector_store.index.ntotal}")

        # Get a sample of documents to see what's in the store
        try:
            # Get all document metadata
            docstore = self.vector_store.docstore
            print(f"Docstore size: {len(docstore._dict)}")

            # Count documents by book
            book_counts = {}
            for doc_id, doc in docstore._dict.items():
                book_name = doc.metadata.get("book", "Unknown")
                book_counts[book_name] = book_counts.get(book_name, 0) + 1

            print(f"\n📚 Documents by book:")
            for book, count in sorted(book_counts.items()):
                print(f"  {book}: {count} chunks")

            # Test embedding quality with a simple query
            print(f"\n🧪 Testing embedding quality:")
            test_queries = [
                "machine learning",
                "neural networks",
                "backpropagation",
                "gradient descent",
            ]

            for query in test_queries:
                docs = self.vector_store.similarity_search_with_score(query, k=3)
                print(f"\nQuery: '{query}'")
                for i, (doc, score) in enumerate(docs):
                    book = doc.metadata.get("book", "Unknown")
                    page = doc.metadata.get("page", "Unknown")
                    content_preview = doc.page_content[:100].replace("\n", " ")
                    print(f"  {i+1}. {book} (Page {page}) - Score: {score:.3f}")
                    print(f"     Content: {content_preview}...")

        except Exception as e:
            print(f"❌ Error during debug: {e}")
            import traceback

            traceback.print_exc()

    def rebuild_vector_store(self):
        """Rebuild the vector store from scratch."""
        print("🔄 Rebuilding vector store from scratch...")

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
            print("✅ Cleared existing vector store")
        except Exception as e:
            print(f"⚠️ Error clearing existing data: {e}")

    def add_multiple_pdfs(self, pdf_paths: List[str], max_workers: int = 2):
        """Add multiple PDFs efficiently.

        Args:
            pdf_paths: List of PDF file paths
            max_workers: Number of parallel workers for PDF text extraction
        """
        print(f"🚀 Processing {len(pdf_paths)} PDFs with {max_workers} workers...")

        # Filter out already processed PDFs
        unprocessed_pdfs = []
        for pdf_path in pdf_paths:
            pdf_name = Path(pdf_path).name
            if pdf_name not in self.processed_pdfs:
                unprocessed_pdfs.append(pdf_path)
            else:
                print(f"📚 Skipping already processed: {pdf_name}")

        if not unprocessed_pdfs:
            print("✅ All PDFs already processed!")
            return

        print(f"📖 Processing {len(unprocessed_pdfs)} new PDFs...")

        # Process each PDF (keep sequential for now to avoid memory issues)
        for pdf_path in unprocessed_pdfs:
            try:
                self.add_pdf(pdf_path)
            except Exception as e:
                print(f"❌ Failed to process {Path(pdf_path).name}: {str(e)}")
                continue

    def add_multiple_pdfs_parallel(self, pdf_paths: List[str], max_workers: int = None):
        """Add multiple PDFs using parallel processing for much faster performance.

        Args:
            pdf_paths: List of PDF file paths
            max_workers: Number of parallel workers (defaults to CPU count)
        """
        if max_workers is None:
            max_workers = min(mp.cpu_count(), len(pdf_paths))

        print(
            f"🚀 Processing {len(pdf_paths)} PDFs with {max_workers} parallel workers..."
        )

        # Filter out already processed PDFs
        unprocessed_pdfs = []
        for pdf_path in pdf_paths:
            pdf_name = Path(pdf_path).name
            if pdf_name not in self.processed_pdfs:
                unprocessed_pdfs.append(pdf_path)
            else:
                print(f"📚 Skipping already processed: {pdf_name}")

        if not unprocessed_pdfs:
            print("✅ All PDFs already processed!")
            return

        print(f"📖 Processing {len(unprocessed_pdfs)} new PDFs in parallel...")

        # Prepare arguments for multiprocessing
        args = [(pdf_path, 512, 50) for pdf_path in unprocessed_pdfs]

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
                        print(f"❌ {pdf_name}: {chunks}")
                        continue

                    print(f"✅ {pdf_name}: {num_pages} pages, {len(chunks)} chunks")

                    # Collect chunks and metadata
                    for chunk in chunks:
                        all_chunks.append(chunk["text"])
                        all_metadatas.append(chunk["metadata"])

                    processed_files.append(pdf_name)

                except Exception as e:
                    print(f"❌ {pdf_name}: {str(e)}")

        if not all_chunks:
            print("⚠️ No valid chunks to add!")
            return

        # Add all chunks to vector store in one go (much faster)
        print(
            f"🔄 Adding {len(all_chunks)} chunks to vector store with GPU acceleration..."
        )

        try:
            if self.vector_store is None:
                print("🆕 Creating new vector store...")
                self.vector_store = FAISS.from_texts(
                    all_chunks, self.embeddings, metadatas=all_metadatas
                )
            else:
                print("➕ Adding to existing vector store...")
                self.vector_store.add_texts(all_chunks, metadatas=all_metadatas)

            # Mark all PDFs as processed
            for pdf_name in processed_files:
                self.processed_pdfs.add(pdf_name)

            self._save_processed_pdfs()
            self._save_vector_store()

            print(
                f"✅ Successfully processed {len(processed_files)} PDFs with {len(all_chunks)} total chunks"
            )

        except Exception as e:
            print(f"❌ Error adding chunks to vector store: {e}")
            raise


def main():
    """Example usage of the PDF RAG system."""
    import sys
    import time

    # Check if user wants to rebuild
    rebuild = "--rebuild" in sys.argv

    # Initialize RAG system with optimized settings
    rag = PDFRAG(max_chunk_size=512, similarity_threshold=0.1)

    if rebuild:
        print("🚀 Fast Embedding Rebuild Mode")
        print("=" * 50)

        # Force rebuild
        print("🔄 Clearing existing embeddings...")
        rag.rebuild_vector_store()

        # Get all PDFs
        pdf_dir = Path("RAG")
        pdf_files = list(pdf_dir.glob("*.pdf"))

        print(f"\n📚 Found {len(pdf_files)} PDF files")
        for i, pdf_file in enumerate(pdf_files, 1):
            size_mb = pdf_file.stat().st_size / (1024 * 1024)
            print(f"  {i}. {pdf_file.name} ({size_mb:.1f} MB)")

        # Process all PDFs in parallel
        print(f"\n🚀 Starting parallel processing with GPU acceleration...")
        start_time = time.time()

        # Use parallel processing for much faster performance
        rag.add_multiple_pdfs_parallel([str(f) for f in pdf_files])

        # Final summary
        total_time = time.time() - start_time
        if rag.vector_store:
            total_vectors = rag.vector_store.index.ntotal
            print(f"\n🎉 Rebuild complete!")
            print(f"⏱️ Total time: {total_time:.1f} seconds")
            print(f"📊 Total vectors: {total_vectors}")
            print(f"🚀 Speed: {total_vectors/total_time:.1f} vectors/second")

            # Quick test
            print(f"\n🧪 Quick test retrieval:")
            test_results = rag.retrieve_documents("machine learning", k=3)
            for doc in test_results:
                print(
                    f"  ✓ {doc['book']} (Page {doc['page']}) - Score: {doc['score']:.3f}"
                )
        else:
            print(f"\n❌ Rebuild failed after {total_time:.1f} seconds")

        return  # Exit after rebuild

    # Normal processing mode
    # Add PDF books efficiently
    pdf_dir = Path("RAG")
    pdf_files = list(pdf_dir.glob("*.pdf"))

    print(f"\n📚 Found {len(pdf_files)} PDF files")

    # Use the optimized batch processing
    rag.add_multiple_pdfs([str(f) for f in pdf_files])

    # Debug: Check vector store status
    if rag.vector_store is not None:
        print(
            f"\n🎯 Vector store status: LOADED with {rag.vector_store.index.ntotal} vectors"
        )
        rag.debug_vector_store()  # Add debug analysis
    else:
        print("\n❌ Vector store status: NOT LOADED")
        return

    # Example questions
    questions = [
        "What is the definition of machine learning?",
        "Explain the concept of backpropagation.",
        "What are the main types of neural networks?",
        "How does gradient descent work?",
    ]

    for question in questions:
        print(f"\n{'='*80}")
        print(f"❓ Question: {question}")
        print("-" * 80)

        # Show retrieved documents with no score threshold for debugging
        print("\n📚 Retrieved Documents (no threshold):")
        retrieved = rag.retrieve_documents(question, k=5, score_threshold=None)
        for i, doc in enumerate(retrieved):
            print(f"\n{i+1}. Book: {doc['book']}")
            print(f"   Page: {doc['page']}")
            print(f"   Score: {doc['score']:.3f}")
            print(f"   Content: {doc['content'][:200]}...")

        if not retrieved:
            print(
                "⚠️ No documents retrieved! There might be an issue with the embeddings."
            )
            continue

        # Generate answer
        result = rag.generate_answer(question, score_threshold=None)

        print(f"\n🎯 Answer:\n{result['answer']}")
        print(f"\n📚 Sources:")
        for source, score in zip(result["sources"], result["retrieval_scores"]):
            print(f"- {source} (Score: {score:.3f})")
        print(f"\n📄 Documents used: {result['context_docs']}")


if __name__ == "__main__":
    main()
