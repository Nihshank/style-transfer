import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class VGG19:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # fetch only the convolutional blocks - discard the classifier 
        self.model = models.vgg19(pretrained=True).features.to(self.device)
        
        for param in self.model.parameters():
            param.requires_grad = False


class ImageProcessor:
    def __init__(self, device, image_size=224):
        self.device = device
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def load_image(self, path):
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        image = image.unsqueeze(0)
        return image.to(self.device)