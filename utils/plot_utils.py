import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Plots the training and validation set losses side by side
def plot_losses(epochs_seen,
                tokens_seen,
                train_losses,
                val_losses):
     
     fig, ax1 = plt.subplots(figsize=(5,3))
     ax1.plot(epochs_seen,
              train_losses,
              label="Training loss")
     ax1.plot(epochs_seen,
              val_losses,
              linestyle="-.",
              label="Validation loss")
     ax1.set_xlabel =("Epochs")
     ax1.set_ylabel("Loss")
     ax1.legend(loc="upper right")
     ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
     ax2 = ax1.twiny()
     ax2.plot(tokens_seen, train_losses, alpha=0)
     ax2.set_xlabel("Tokens seen")
     fig.tight_layout()
     
     plt.show()

# Plots classification loss
def plot_values(epochs_seen,
                examples_seen,
                train_values,
                val_values,
                label="loss"):
     
     fig, ax1 = plt.subplots(figsize=(5,3))
     ax1.plot(epochs_seen,
              train_values,
              label="Training {label}")
     ax1.plot(epochs_seen,
              val_values,
              linestyle="-.",
              label="Validation {label}")
     ax1.set_xlabel =("Epochs")
     ax1.set_ylabel(label.capitalize())
     ax1.legend()

     # Create a second x-axis for examples seen
     ax2 = ax1.twiny()

     # Invisible plot for aligning ticks
     ax2.plot(examples_seen, 
              train_values, 
              alpha=0)
     ax2.set_xlabel("Examples seen")

     # Adjust layout to make room
     fig.tight_layout()
     plt.savefig(f"{label}-plot.pdf")
     plt.show()
