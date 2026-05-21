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
    
def get_full_image_base_transform(input_size: int):
    """
    給單階段全圖資料集使用的 base transform
    （不含隨機 augmentation）
    注意：這裡不做 Resize，因為 Dataset 已經做好 Resize 了
    """
    return transforms.Compose([
        transforms.ToTensor(),
    ])