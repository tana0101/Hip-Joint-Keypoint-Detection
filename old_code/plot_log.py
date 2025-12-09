import os
import re
import matplotlib.pyplot as plt

def parse_log_file(file_path):
    """
    解析單一日誌文件，提取 Loss、NME 和 Pixel Error 資料。
    Args:
        file_path: 日誌檔案路徑。
    Returns:
        epochs: List of epoch numbers
        epoch_losses: List of training losses
        val_losses: List of validation losses
        epoch_nmes: List of training NMEs
        val_nmes: List of validation NMEs
        epoch_pixel_errors: List of training pixel errors
        val_pixel_errors: List of validation pixel errors
    """
    epochs = []
    epoch_losses = []
    val_losses = []
    epoch_nmes = []
    val_nmes = []
    epoch_pixel_errors = []
    val_pixel_errors = []

    # 正則表達式匹配每行數據
    pattern = re.compile(
        r"Epoch (\d+): Loss = ([\d\.]+), NME = ([\d\.]+), Pixel Error = ([\d\.]+), "
        r"Val Loss = ([\d\.]+), Val NME = ([\d\.]+), Val Pixel Error = ([\d\.]+)"
    )

    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                epochs.append(int(match.group(1)))
                epoch_losses.append(float(match.group(2)))
                epoch_nmes.append(float(match.group(3)))
                epoch_pixel_errors.append(float(match.group(4)))
                val_losses.append(float(match.group(5)))
                val_nmes.append(float(match.group(6)))
                val_pixel_errors.append(float(match.group(7)))
    
    return epochs, epoch_losses, val_losses, epoch_nmes, val_nmes, epoch_pixel_errors, val_pixel_errors


def plot_training_progress(epochs_range, epoch_losses, val_losses, epoch_nmes, val_nmes, epoch_pixel_errors, val_pixel_errors, 
                           title_suffix="", start_epoch=1, loss_ylim=None, nme_ylim=None, pixel_error_ylim=None, save_path=None):
    """
    Function to plot training and validation progress for Loss, NME, and Pixel Error.
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
    
    if save_path:
        plt.savefig(save_path)
        print(f"圖表已儲存至: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main(log_dir):
    """
    主函數，讀取資料夾內所有txt檔案並將圖檔存至 log_directory。
    Args:
        log_dir: 資料夾路徑
    """
    all_files = [f for f in os.listdir(log_dir) if f.endswith('.txt')]

    for file in all_files:
        file_path = os.path.join(log_dir, file)
        print(f"正在處理檔案: {file_path}")
        
        # 解析日誌檔案
        epochs, epoch_losses, val_losses, epoch_nmes, val_nmes, epoch_pixel_errors, val_pixel_errors = parse_log_file(file_path)
        
        # 儲存圖檔
        plot_save_path = os.path.join(log_dir, f"plot_{os.path.splitext(file)[0]}.png")
        plot_training_progress(
            epochs_range=epochs,
            epoch_losses=epoch_losses,
            val_losses=val_losses,
            epoch_nmes=epoch_nmes,
            val_nmes=val_nmes,
            epoch_pixel_errors=epoch_pixel_errors,
            val_pixel_errors=val_pixel_errors,
            title_suffix=f" ({file})",
            loss_ylim=(0, 300),
            nme_ylim=(0, 0.02),
            pixel_error_ylim=(0, 200),
            save_path=plot_save_path
        )



if __name__ == "__main__":
    log_directory = "./results/04/original/efficientnet_1000_0.01_32"  # 修改為你的資料夾路徑
    main(log_directory)
