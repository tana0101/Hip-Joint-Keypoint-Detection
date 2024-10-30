import os
import re
import matplotlib.pyplot as plt

def load_training_logs(folder_path):
    log_data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith("_training_log.txt"):
            model_name = filename[:-17]  
            log_data[model_name] = {'epoch': [], 'loss': [], 'nme': [], 'pixel_error': []}
            
            with open(os.path.join(folder_path, filename), 'r') as file:
                for line in file:
                    match = re.search(r"Epoch (\d+): Loss = ([\d\.]+), NME = ([\d\.]+), Pixel Error = ([\d\.]+)", line)
                    if match:
                        epoch = int(match.group(1))
                        loss = float(match.group(2))
                        nme = float(match.group(3))
                        pixel_error = float(match.group(4))
                        
                        log_data[model_name]['epoch'].append(epoch)
                        log_data[model_name]['loss'].append(loss)
                        log_data[model_name]['nme'].append(nme)
                        log_data[model_name]['pixel_error'].append(pixel_error)

    return log_data

def plot_training_logs(log_data, output_folder):
    plt.figure(figsize=(18, 6))  # Adjusted size for better display

    # Plotting Loss
    plt.subplot(1, 3, 1)  # 1 row, 3 columns, 1st subplot
    for model_name, metrics in log_data.items():
        epochs = metrics['epoch']
        plt.plot(epochs, metrics['loss'], label=model_name, marker='o')
    
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')  
    plt.legend()
    plt.grid()

    # Plotting NME
    plt.subplot(1, 3, 2)  # 1 row, 3 columns, 2nd subplot
    for model_name, metrics in log_data.items():
        epochs = metrics['epoch']
        plt.plot(epochs, metrics['nme'], label=model_name, marker='o')
    
    plt.title('Normalized Mean Error (NME)')
    plt.xlabel('Epoch')
    plt.ylabel('NME')
    plt.yscale('log')  
    plt.legend()
    plt.grid()

    # Plotting Pixel Error
    plt.subplot(1, 3, 3)  # 1 row, 3 columns, 3rd subplot
    for model_name, metrics in log_data.items():
        epochs = metrics['epoch']
        plt.plot(epochs, metrics['pixel_error'], label=model_name, marker='o')
    
    plt.title('Pixel Error')
    plt.xlabel('Epoch')
    plt.ylabel('Pixel Error')
    plt.yscale('log')  
    plt.legend()
    plt.grid()

    plt.tight_layout()
    
    output_path = os.path.join(output_folder, 'training_logs_comparison.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    folder_path = "logs"  
    log_data = load_training_logs(folder_path)
    plot_training_logs(log_data, folder_path)
