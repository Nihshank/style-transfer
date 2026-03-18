import torch
import torchvision.models as models

class VGG19:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # fetch only the convolutional blocks - discard the classifier 
        self.model = models.vgg19(pretrained=True).features.to(self.device)
        
        for param in self.model.parameters():
            param.requires_grad = False

