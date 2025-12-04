import torch.nn as nn
import matplotlib.pyplot as plt

class GraphWrapper(nn.Module):
    def __init__(self, model, head_type):
        super().__init__()
        self.model = model
        self.head_type = head_type

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, dict):
            if self.head_type == "direct_regression":
                return out["coords"]       # [B, J, 2]
            else:
                return out["logits_x"]     # simcc 系列就挑一個輸出給 tracer
        return out
    
def display_image(dataset, index):
    # Get the nth image and keypoints
    image, keypoints, original_size = dataset[index]
    print(f"Displaying image {index}")
    print("Original size:", original_size)
    print("Image shape:", image.shape)
    
    # Convert the image to a NumPy array
    image_np = image.permute(1, 2, 0).numpy()

    # Display the image
    plt.imshow(image_np, cmap='gray')
    plt.title(f"Image {index} with Keypoints")
    
    # Plot the keypoints with numbering
    for i in range(0, len(keypoints), 2):
        x = keypoints[i].item()  
        y = keypoints[i + 1].item()  
        plt.scatter(x, y, c='red', s=20)
        plt.text(x, y, f'{i//2 + 1}', color='yellow', fontsize=12)  # Add number next to each point

    plt.show()

def plot_training_progress(epochs_range, epoch_losses, val_losses, epoch_nmes, val_nmes, epoch_pixel_errors, val_pixel_errors, 
                           title_suffix="", start_epoch=1, loss_ylim=None, nme_ylim=None, pixel_error_ylim=None):
    """
    Function to plot training and validation progress for Loss, NME, and Pixel Error.
    Args:
        epochs_range: Range of epochs to plot
        epoch_losses: Training losses for each epoch
        val_losses: Validation losses for each epoch
        epoch_nmes: Training NME for each epoch
        val_nmes: Validation NME for each epoch
        epoch_pixel_errors: Training Pixel Error for each epoch
        val_pixel_errors: Validation Pixel Error for each epoch
        title_suffix: Optional suffix for the plot titles (e.g., " (Epoch 20 onwards)")
        start_epoch: Epoch to start plotting from (default is 1, to plot from the start)
        loss_ylim: Tuple for Loss y-axis limits (e.g., (0, 1))
        nme_ylim: Tuple for NME y-axis limits (e.g., (0, 0.1))
        pixel_error_ylim: Tuple for Pixel Error y-axis limits (e.g., (0, 5))
    """
    # Extract data from start_epoch
    if start_epoch > 1:
        epochs_range = range(start_epoch, len(epoch_losses) + 1)
        epoch_losses = epoch_losses[start_epoch - 1:]
        epoch_nmes = epoch_nmes[start_epoch - 1:]
        epoch_pixel_errors = epoch_pixel_errors[start_epoch - 1:]
        val_losses = val_losses[start_epoch - 1:]
        val_nmes = val_nmes[start_epoch - 1:]
        val_pixel_errors = val_pixel_errors[start_epoch - 1:]

    plt.figure(figsize=(12, 6))

    # Plot Loss with log scale
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, epoch_losses, label="Training Loss")
    plt.plot(epochs_range, val_losses, label="Validation Loss")
    plt.title(f"Loss{title_suffix}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log)")
    plt.yscale('log')  # Log scale
    if loss_ylim:
        plt.ylim(loss_ylim)  # Set y-axis limits for Loss
    plt.legend()

    # Plot NME
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, epoch_nmes, label="Training NME")
    plt.plot(epochs_range, val_nmes, label="Validation NME")
    plt.title(f"NME{title_suffix}")
    plt.xlabel("Epoch")
    plt.ylabel("NME (log)")
    plt.yscale('log')  # Log scale
    if nme_ylim:
        plt.ylim(nme_ylim)  # Set y-axis limits for NME
    plt.legend()

    # Plot Pixel Error
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, epoch_pixel_errors, label="Training Pixel Error")
    plt.plot(epochs_range, val_pixel_errors, label="Validation Pixel Error")
    plt.title(f"Pixel Error{title_suffix}")
    plt.xlabel("Epoch")
    plt.ylabel("Pixel Error (log)")
    plt.yscale('log')  # Log scale
    if pixel_error_ylim:
        plt.ylim(pixel_error_ylim)  # Set y-axis limits for Pixel Error
    plt.legend()

    plt.tight_layout()