#!/usr/bin/env python3
"""
PDF utilities for text extraction and processing.
Contains PDF handling functions for the RAG system.

Performance optimizations included:
- PyMuPDF (fitz) for ultra-fast PDF text extraction (5-10x faster than PyPDF2)
- CUDA-free processing to avoid multiprocessing issues
- Optimized memory usage and error handling
- Automatic fallback to PyPDF2 if PyMuPDF is not available

Install PyMuPDF for maximum performance:
    pip install PyMuPDF

If PyMuPDF is not available, the system automatically falls back to PyPDF2.

This module follows a pure extraction approach:
- Only handles PDF text extraction, no chunking or processing
- No knowledge of RAG-specific requirements
- Clean separation of concerns with rag_utils.py
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import PyPDF2

logger = logging.getLogger(__name__)


def check_pymupdf_installation():
    """Check if PyMuPDF is installed and log performance recommendations."""
    try:
        import fitz
        logger.info("✅ PyMuPDF (fitz) detected - using ultra-fast PDF processing")
        return True
    except ImportError:
        logger.warning("⚠️  PyMuPDF not found - falling back to PyPDF2 (slower)")
        logger.warning("   For 5-10x faster PDF processing, install: pip install PyMuPDF")
        return False


def extract_text_from_single_pdf(pdf_path: Path, include_metadata: bool = True) -> Union[List[Dict[str, Any]], List[str]]:
    """Extract text from a single PDF with optional metadata using PyMuPDF for optimal performance.

    Responsibility: Pure text extraction from one PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        include_metadata: Whether to include page metadata
            - True: Returns [{"text": "...", "metadata": {...}}, ...]
            - False: Returns ["page_text_1", "page_text_2", ...]

    Returns:
        List of dictionaries containing text and optionally metadata
    """
    logger.debug(f"Extracting text from PDF: {pdf_path}")

    # Get book title from filename
    book_title = pdf_path.stem
    documents = []
    
    try:
        import fitz  # PyMuPDF
        
        # Open PDF with fitz (PyMuPDF) - faster than PyPDF2
        pdf_document = fitz.open(str(pdf_path))
        num_pages = len(pdf_document)

        # Extract text from each page
        for page_num in range(num_pages):
            page = pdf_document[page_num]
            # Extract text using fitz (faster and more accurate)
            text = page.get_text("text")

            if text and text.strip():  # Only add non-empty pages
                if include_metadata:
                    documents.append({
                        "text": text.strip(),
                        "metadata": {
                            "book": book_title,
                            "page": page_num + 1,
                            "total_pages": num_pages,
                        },
                    })
                else:
                    documents.append(text.strip())
        
        # Close the PDF document to free memory
        pdf_document.close()
        
    except ImportError:
        # Fallback to PyPDF2 if PyMuPDF is not available
        logger.warning("PyMuPDF not available, falling back to PyPDF2 (slower)")
        
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)

            # Extract text from each page
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text = page.extract_text()

                if text and text.strip():  # Only add non-empty pages
                    if include_metadata:
                        documents.append({
                            "text": text.strip(),
                            "metadata": {
                                "book": book_title,
                                "page": page_num + 1,
                                "total_pages": num_pages,
                            },
                        })
                    else:
                        documents.append(text.strip())

    return documents


def extract_text_from_multiple_pdfs(
    pdf_paths: List[Path], 
    include_metadata: bool = True, 
    max_workers: Optional[int] = None
) -> Union[List[Dict[str, Any]], List[str]]:
    """Extract text from multiple PDFs, potentially in parallel.

    Responsibility: Pure text extraction from multiple PDF files with parallelization.
    
    Args:
        pdf_paths: List of PDF file paths to process
        include_metadata: Whether to include metadata in the output
        max_workers: Number of parallel workers (defaults to CPU count)

    Returns:
        Flattened list of all texts/data from all PDFs in the same format as extract_text_from_single_pdf
    """
    if not pdf_paths:
        return []
    
    if max_workers is None:
        import multiprocessing as mp
        max_workers = min(len(pdf_paths), mp.cpu_count())
    
    logger.info(f"Extracting text from {len(pdf_paths)} PDFs with {max_workers} parallel workers (CUDA-free)")
    
    start_time = time.time()
    all_results = []
    
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all PDF processing jobs
            future_to_path = {
                executor.submit(_extraction_worker, pdf_path, include_metadata): pdf_path
                for pdf_path in pdf_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_path):
                pdf_path = future_to_path[future]
                try:
                    results = future.result()
                    if results:
                        all_results.extend(results)
                        logger.debug(f"✅ Processed {pdf_path.name}: {len(results)} pages")
                    else:
                        logger.warning(f"⚠️  No text extracted from {pdf_path.name}")
                        
                except Exception as e:
                    logger.error(f"❌ Error processing {pdf_path.name}: {str(e)}")
                    continue
        
        total_time = time.time() - start_time
        logger.info(f"🚀 Parallel text extraction completed!")
        logger.info(f"📊 Total pages extracted: {len(all_results)}")
        logger.info(f"⏱️  Total time: {total_time:.2f} seconds")
        if total_time > 0:
            logger.info(f"🏃 Speed: {len(all_results)/total_time:.2f} pages/second (parallel)")
        
        return all_results
        
    except Exception as e:
        logger.error(f"Error in parallel processing: {e}")
        # Fallback to sequential processing
        logger.info("Falling back to sequential processing...")
        for pdf_path in pdf_paths:
            try:
                results = extract_text_from_single_pdf(pdf_path, include_metadata)
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Error processing {pdf_path.name}: {e}")
                continue
        
        return all_results


def _extraction_worker(pdf_path: Path, include_metadata: bool) -> Union[List[Dict[str, Any]], List[str]]:
    """Worker function for parallel PDF text extraction.
    
    Responsibility: CUDA-safe worker for extract_text_from_multiple_pdfs.
    
    Args:
        pdf_path: Path to the PDF file
        include_metadata: Whether to include metadata
        
    Returns:
        List of extracted text data from the PDF
    """
    # Ensure CUDA is not available in worker processes to avoid multiprocessing issues
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    try:
        return extract_text_from_single_pdf(pdf_path, include_metadata)
    except Exception as e:
        logger.error(f"Worker error processing {pdf_path.name}: {str(e)}")
        return []


def find_rag_directory(project_root: Path, rag_dir: Path, results_dir: Path) -> Path:
    """Find and return the RAG directory path with fallback logic.
    
    Args:
        project_root: Project root directory
        rag_dir: Preferred RAG directory path
        results_dir: Results directory path
        
    Returns:
        Path to the RAG directory
    """
    # Priority order for RAG directory discovery:
    # 1. Explicitly provided rag_dir
    # 2. project_root/RAG
    # 3. project_root/rag  
    # 4. results_dir/RAG
    # 5. results_dir/rag
    
    candidate_paths = []
    
    if rag_dir is not None:
        candidate_paths.append(rag_dir)
    
    if project_root is not None:
        candidate_paths.extend([
            project_root / "RAG",
            project_root / "rag"
        ])
    
    if results_dir is not None:
        candidate_paths.extend([
            results_dir / "RAG", 
            results_dir / "rag"
        ])
    
    # Try to find existing directory
    for path in candidate_paths:
        if path.exists() and path.is_dir():
            logger.info(f"Found existing RAG directory: {path}")
            return path
    
    # If no existing directory found, create the first candidate
    if candidate_paths:
        chosen_path = candidate_paths[0]
        chosen_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created RAG directory: {chosen_path}")
        return chosen_path
    
    # Last resort: create RAG in current directory
    fallback_path = Path("RAG")
    fallback_path.mkdir(exist_ok=True)
    logger.warning(f"Created fallback RAG directory: {fallback_path.absolute()}")
    return fallback_path 