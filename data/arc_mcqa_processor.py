from datasets import load_dataset, Dataset, DatasetDict
from typing import List, Dict, Tuple
import logging
from pathlib import Path
import sys

# Importar el base processor correctamente
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent))
    from base_processor import BaseDatasetProcessor
else:
    from .base_processor import BaseDatasetProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ARCProcessor(BaseDatasetProcessor):
    """Processor for the AI2 ARC dataset (Challenge and Easy)."""

    def __init__(self, subset: str = "ARC-Challenge", *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert subset in ["ARC-Challenge", "ARC-Easy", "ARC-Merged"], "Subset must be 'ARC-Challenge', 'ARC-Easy', or 'ARC-Merged'"
        self.subset = subset
        if subset == "ARC-Merged":
            self.source_name = "ai2_arc_merged"
        else:
            self.source_name = f"ai2_arc_{'challenge' if subset == 'ARC-Challenge' else 'easy'}"

    def process_dataset(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        logger.info(f"Loading AI2 ARC dataset ({self.subset})...")
        dataset = load_dataset("allenai/ai2_arc", self.subset)
        logger.info(f"Loaded splits: {list(dataset.keys())}")

        train_data = self._process_split(dataset.get("train", []), split_name="train")
        val_data = self._process_split(dataset.get("validation", []), split_name="validation")
        test_data = self._process_split(dataset.get("test", []), split_name="test")
        return train_data, val_data, test_data

    def _process_split(self, split, split_name: str) -> List[Dict]:
        logger.info(f"Processing {split_name} split ({len(split)} examples)...")
        processed = []
        for i, item in enumerate(split):
            if i % 100 == 0:
                logger.info(f"Processed {i} {split_name} examples")
            processed_item = self._process_item(item)
            if processed_item:
                processed.append(processed_item)
        return processed

    def _process_item(self, item: Dict) -> Dict:
        try:
            question = item["question"].strip()
            choices_text = item["choices"]["text"]
            choices_label = item["choices"]["label"]
            # Formatear las opciones como 'A. opciónA'
            formatted_choices = [f"{label}. {text}" for label, text in zip(choices_label, choices_text)]
            # Buscar el índice de la respuesta correcta
            answer_letter = item["answerKey"].strip()
            try:
                answer_index = choices_label.index(answer_letter)
            except ValueError:
                logger.warning(f"Answer letter {answer_letter} not in labels {choices_label}")
                return None
            answer_text = formatted_choices[answer_index]
            return {
                "question": question,
                "choices": formatted_choices,
                "answer_index": answer_index,
                "answer_text": answer_text,
                "source": self.source_name,
                "explanation": ""
            }
        except Exception as e:
            logger.warning(f"Error processing item: {e}\nRaw item: {item}")
            return None

def merge_datasets(processors: List[ARCProcessor]) -> DatasetDict:
    """Merge multiple ARC datasets into a single dataset."""
    logger.info("Merging ARC datasets...")
    all_train = []
    all_val = []
    all_test = []
    
    for processor in processors:
        train, val, test = processor.process_dataset()
        all_train.extend(train)
        all_val.extend(val)
        all_test.extend(test)
    
    # Create a new processor for the merged dataset
    merged_processor = ARCProcessor(subset="ARC-Merged", output_dir="output/arc_merged")
    return merged_processor.create_dataset_dict(all_train, all_val, all_test)

def main():
    repo_names = {
        "ARC-Challenge": "RikoteMaster/arc_challenge_mcqa",
        "ARC-Easy": "RikoteMaster/arc_easy_mcqa",
        "ARC-Merged": "RikoteMaster/arc_merged_mcqa"
    }
    
    # Process individual datasets
    processors = []
    for subset in ["ARC-Challenge", "ARC-Easy"]:
        processor = ARCProcessor(subset=subset, output_dir=f"output/arc_{subset.lower()}")
        processor.process_and_save()
        processors.append(processor)
        
        # Push individual datasets to Hugging Face Hub
        repo_name = repo_names[subset]
        try:
            from dotenv import load_dotenv
            import os
            load_dotenv()
            token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
            if not token:
                logger.warning("No Hugging Face token found in environment. Skipping push to hub.")
                continue
            train_data, val_data, test_data = processor.process_dataset()
            dataset_dict = processor.create_dataset_dict(train_data, val_data, test_data)
            logger.info(f"Pushing {subset} to Hugging Face Hub at {repo_name} ...")
            dataset_dict.push_to_hub(repo_name, token=token)
            logger.info(f"✅ Successfully pushed {subset} to Hugging Face Hub!")
        except Exception as e:
            logger.error(f"Failed to push {subset} to hub: {e}")
    
    # Merge and push combined dataset
    try:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if not token:
            logger.warning("No Hugging Face token found in environment. Skipping push of merged dataset to hub.")
        else:
            merged_dataset = merge_datasets(processors)
            repo_name = repo_names["ARC-Merged"]
            logger.info(f"Pushing merged ARC dataset to Hugging Face Hub at {repo_name} ...")
            merged_dataset.push_to_hub(repo_name, token=token)
            logger.info("✅ Successfully pushed merged ARC dataset to Hugging Face Hub!")
    except Exception as e:
        logger.error(f"Failed to push merged dataset to hub: {e}")

    
if __name__ == "__main__":
    main() 