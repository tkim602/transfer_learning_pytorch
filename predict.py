import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

def load_model(model_path, num_classes):
    """
    Load the saved model weights.
    """
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

def get_transforms():
    """
    Define the transformations for preprocessing the image.
    """
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return data_transforms

def predict_image(image_path, model, transform, class_names):
    """
    Perform prediction on a single image.
    """
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add a batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    return class_names[preds[0]]

if __name__ == '__main__':
    # Define class names (must match the classes used during training)
    class_names = ['ants', 'bees']
    
    model_path = 'prediction_model.pth'
    model = load_model(model_path, num_classes=len(class_names))
    transform = get_transforms()
    image_path = 'bee_test.jpg'
    predicted_class = predict_image(image_path, model, transform, class_names)
    print(f'Predicted class: {predicted_class}')

