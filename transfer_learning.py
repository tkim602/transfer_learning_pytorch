import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import sys

def main():
    # Enable CUDNN benchmark for optimized performance
    cudnn.benchmark = True
    plt.ion()  # Enable interactive mode for plotting

    # Define data directories using relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory where the script is located
    data_dir = os.path.join(script_dir, 'dataset')
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    # Check if train and val directories exist
    for directory in [train_dir, val_dir]:
        if not os.path.exists(directory):
            print(f"Error: Directory '{directory}' does not exist.")
            sys.exit(1)

    # Define data augmentation and normalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomAffine(degrees=20, translate=(0.2, 0.2)),
            transforms.RandomHorizontalFlip(p=0.6),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
            transforms.RandomRotation(25),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
    }

    # Initialize datasets
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transforms['train']),
        'val': datasets.ImageFolder(val_dir, data_transforms['val'])
    }

    # Initialize dataloaders with num_workers=0 for debugging
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=16, shuffle=True, num_workers=0),
        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=16, shuffle=False, num_workers=0)
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS backend is available. Using GPU (MPS).")
    else:
        device = torch.device("cpu")
        print("GPU not available. Using CPU.")

    # Function to display a batch of images
    def imshow(inp, title=None):
        """Display a tensor as an image."""
        inp = inp.numpy().transpose((1, 2, 0))  # Convert from Tensor image
        mean = np.array([0.485, 0.456, 0.406])  # Mean for normalization
        std = np.array([0.229, 0.224, 0.225])   # Std for normalization
        inp = std * inp + mean  # Unnormalize
        inp = np.clip(inp, 0, 1)  # Clip to valid range
        plt.imshow(inp)
        if title:
            plt.title(title)
        plt.pause(0.001)  # Pause to update plots

    # Get a batch of training data
    try:
        inputs, classes_batch = next(iter(dataloaders['train']))
    except Exception as e:
        print(f"Error while loading data: {e}")
        sys.exit(1)

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes_batch])

    # Define the training function
    def train_model(model, criterion, optimizer, scheduler, num_epochs=20):
        """Train and evaluate the model."""
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-'*30)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluation mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data
                for inputs_batch, labels in dataloaders[phase]:
                    inputs_batch = inputs_batch.to(device)
                    labels = labels.to(device)

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs_batch)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # Backward pass and optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # Statistics
                    running_loss += loss.item() * inputs_batch.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()  # Update learning rate

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

                # Deep copy the model if it has better accuracy
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(f'Training completed in {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s')
        print(f'Best Validation Accuracy: {best_acc:.4f}')

        # Load best model weights
        model.load_state_dict(best_model_wts)
        return model

    # Initialize a pre-trained model (EfficientNet-B0)
    model_ft = models.efficientnet_b0(weights=None)
    num_ftrs = model_ft.classifier[1].in_features
    model_ft.classifier[1] = nn.Linear(num_ftrs, len(class_names))

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Define optimizer and scheduler
    optimizer_ft = optim.Adam(model_ft.classifier[1].parameters(), lr=1e-3)
    scheduler_ft = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # Train and evaluate the model
    model_ft = train_model(model_ft, criterion, optimizer_ft, scheduler_ft, num_epochs=20)

    # Save the best model weights
    torch.save(model_ft.state_dict(), 'prediction_model.pth')

    # Visualize predictions
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()
