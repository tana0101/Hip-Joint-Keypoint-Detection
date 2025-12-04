from torchvision import transforms
from PIL import ImageOps

def get_hip_base_transform(input_size: int):
    """
    給 train / val / test / predict 共用的 base transform
    （不含隨機 augmentation）
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Lambda(lambda img: ImageOps.equalize(img)),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])