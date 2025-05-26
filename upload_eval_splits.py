#!/usr/bin/env python3
"""
Script to upload validation and test splits to the sciq_treated_epfl_mcqa repository.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path so we can import data modules
sys.path.append(str(Path(__file__).parent))

from data.sciq_processor import SciQProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Get HF token from environment variable
    hf_token = os.environ.get("HF_TOKEN")

    if not hf_token:
        logger.error(
            "Please set HF_TOKEN environment variable with your Hugging Face token"
        )
        logger.info("You can get a token from: https://huggingface.co/settings/tokens")
        sys.exit(1)

    processor = SciQProcessor()

    try:
        # Push all splits (train, validation, test) to the repository
        processor.push_to_hub(
            repo_name="RikoteMaster/sciq_treated_epfl_mcqa", token=hf_token
        )

        logger.info("🎉 Successfully uploaded all splits!")
        logger.info("📊 The repository now includes:")
        logger.info("   - train split")
        logger.info("   - validation split")
        logger.info("   - test split")
        logger.info("")
        logger.info(
            "🔗 View at: https://huggingface.co/datasets/RikoteMaster/sciq_treated_epfl_mcqa"
        )

    except Exception as e:
        logger.error(f"❌ Failed to upload: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
