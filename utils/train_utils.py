"""Training utilities for model training and visualization."""

import matplotlib.pyplot as plt
from pathlib import Path
from transformers import Trainer

def plot_training_loss(trainer: Trainer, figs_dir: Path) -> None:
    """Plot and save the training loss curve.
    
    Args:
        trainer: The trained Trainer instance containing loss history
        figs_dir: Directory where to save the plot
    """
    loss_history = [log["loss"] for log in trainer.state.log_history if "loss" in log]

    plt.figure(figsize=(6, 4))
    plt.plot(range(len(loss_history)), loss_history, marker='o', label="Training loss")
    plt.xlabel("Log Step")
    plt.ylabel("Loss")
    plt.title("Loss during SFT training")
    plt.legend()
    plt.grid(True)
    
    # Ensure figs directory exists and save the figure
    figs_dir.mkdir(exist_ok=True)
    plt.savefig(figs_dir / "loss_curve.png")
    plt.show() 


def train_open_answer(model, tokenizer, dataset): 

    """This code is used to train qwen model with an open answer dataset, by this way, we can improve the model's reasoning ability
    
    Args:
        model: The model to train
        tokenizer: The tokenizer to use
        dataset: The dataset to train on

    Returns:
        The trained model
    """
    



