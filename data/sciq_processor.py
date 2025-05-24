from datasets import load_dataset
from typing import List, Dict, Tuple
from .base_processor import BaseDatasetProcessor

class SciQProcessor(BaseDatasetProcessor):
    """Processor for the SciQ dataset."""
    
    def process_dataset(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Process the SciQ dataset into MCQA format.
        
        Returns:
            Tuple[List[Dict], List[Dict], List[Dict]]: Training, validation, and test data in MCQA format
        """
        # Load the SciQ dataset
        dataset = load_dataset("sciq")
        
        train_format = []
        val_format = []
        test_format = []
        
        # Process training data
        for item in dataset["train"]:
            processed_item = self._process_item(item)
            if processed_item:
                train_format.append(processed_item)
        
        # Process validation data
        for item in dataset["validation"]:
            processed_item = self._process_item(item)
            if processed_item:
                val_format.append(processed_item)
        
        # Process test data
        for item in dataset["test"]:
            processed_item = self._process_item(item)
            if processed_item:
                test_format.append(processed_item)
        
        return train_format, val_format, test_format
    
    def _process_item(self, item: Dict) -> Dict:
        """
        Process a single item from the SciQ dataset.
        
        Args:
            item: Dictionary containing the item data
            
        Returns:
            Dict: Processed item in MCQA format or None if invalid
        """
        question = item["question"]
        correct = item["correct_answer"]
        distractors = [item["distractor1"], item["distractor2"], item["distractor3"]]
        
        # Skip examples with fewer than 3 distractors
        if len(distractors) != 3:
            return None
        
        # Combine correct answer and distractors
        choices = distractors + [correct]
        choices = list(set(choices))  # deduplicate
        
        # Skip if we don't have exactly 4 unique choices
        if len(choices) != 4:
            return None
        
        # Sort choices for deterministic order
        choices = sorted(choices)
        
        try:
            answer_index = choices.index(correct)
        except ValueError:
            return None  # correct answer not found
        
        # Create MCQA format example
        return {
            "question": question,
            "choices": choices,
            "answer_index": answer_index,
            "answer_text": choices[answer_index],
            "source": "sciq",
            "explanation": item["support"]
        }

def main():
    processor = SciQProcessor()
    processor.process_and_save()

if __name__ == "__main__":
    main() 