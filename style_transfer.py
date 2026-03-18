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
    
class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.layers = {
            '0':  'block1_conv1',
            '5':  'block2_conv1',
            '10': 'block3_conv1',
            '19': 'block4_conv1',
            '21': 'block4_conv2', # content feature map (rest for style)
            '28': 'block5_conv1',
        }

    def get_features(self, image):
        features = {}
        x = image # start with the content image

        for name, layer in self.model._modules.items():
            x = layer(x) # process image applying each layer 
            if name in self.layers:
                features[self.layers[name]] = x

        return features


'''
gram matrix for style image - flatten height and width into 1 dimension
not concerned with spatial position; only with textures(patterns) appearing 
together across the image and the strength of such textures
'''
def gram_matrix(tensor):
    _, channels, height, width = tensor.shape
    tensor = tensor.view(channels, height * width)
    gram = torch.mm(tensor, tensor.t())
    return gram


class StyleTransfer:
    def __init__(self, content_path, style_path, image_size=224, content_weight=1, style_weight=1e6):
        self.vgg = VGG19()
        self.device = self.vgg.device
        self.processor = ImageProcessor(self.device, image_size)
        self.extractor = FeatureExtractor(self.vgg.model)
        
        # load images
        self.content_image = self.processor.load_image(content_path)
        self.style_image = self.processor.load_image(style_path)
        
        # extract and store targets once
        self.content_features = self.extractor.get_features(self.content_image)
        self.style_features = self.extractor.get_features(self.style_image)
        
        # compute gram matrices for style image once
        self.style_grams = {
            layer: gram_matrix(self.style_features[layer])
            for layer in self.style_features
        }
        
        self.content_weight = content_weight
        self.style_weight = style_weight

    def compute_losses(self, generated_features):
        # content loss - MSE between generated and content feature maps at block4_conv2
        content_loss = torch.mean(
            (generated_features['block4_conv2'] - self.content_features['block4_conv2']) ** 2
        )

        # style loss - compare gram matrices across all style layers
        style_loss = 0
        for layer in self.style_grams:
            generated_gram = gram_matrix(generated_features[layer])
            style_gram = self.style_grams[layer]
            
            _, channels, height, width = generated_features[layer].shape
            layer_style_loss = torch.mean((generated_gram - style_gram) ** 2)
            style_loss += layer_style_loss / (channels * height * width)

        total_loss = (self.content_weight * content_loss) + (self.style_weight * style_loss)
        return total_loss
    
    
    def optimize(self, steps=5000, save_every=250):
        # content image as starting point
        generated = self.content_image.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([generated], lr=0.003)
        
        for step in range(steps):
            # extract features of generated image
            generated_features = self.extractor.get_features(generated)
        
            # calculate loss
            total_loss = self.compute_losses(generated_features)
            
            # backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if step % save_every == 0:
                print(f'Step {step}, Loss: {total_loss.item():.2f}')
                self.save_image(generated, f'output/step_{step}.png')

