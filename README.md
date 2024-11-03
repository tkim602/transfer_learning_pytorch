# Transfer Learning Project

## Project Overview
This project leverages transfer learning with PyTorch's EfficientNet-B0 model to classify images of ants and bees. Transfer learning allows us to adapt pre-trained models, which have been trained on large datasets, to solve more specific tasks with smaller datasets and less computational effort. Here, we build a pipeline to fine-tune the model on custom classes (ants and bees) and evaluate its performance on unseen data.

---

## Table of Contents
- [Background](#background)
- [Setup](#setup)
- [Training the Model](#training-the-model)
- [Using the Model for Predictions](#using-the-model-for-predictions)
- [Requirements](#requirements)
- [Theory: Transfer Learning](#theory-transfer-learning)


---

## Background
The goal of this project is to implement an image classification system by fine-tuning a pre-trained neural network. Image classification tasks require a model that can recognize patterns in image data, which is often computationally expensive. Instead of training a model from scratch, transfer learning helps by starting with a model pre-trained on a large dataset, like ImageNet, and adapting it to classify different categories (in this case, ants and bees). This approach saves both time and resources while maintaining high accuracy.

---

## Setup

### Prerequisites
Ensure you have Python 3.11 installed and clone this repository. All dependencies are managed in a virtual environment (`myenv_py311`).

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your_username/transfer_learning_project.git
   cd transfer_learning_project
2. Set Up Vitrual Enviroment
   python3 -m venv myenv
   source myenv/bin/activate
3. Install Dependencies
   pip install -r requirements.txt

---

## Train the Model

1. Organize the Dataset

   ![image](https://github.com/user-attachments/assets/13d421ce-e237-4b32-8a2e-b01aa2896f6e)


4. Run the Training Script
   python3 transfer_learning.py
   -> EfficientNet-B0 model will be trained using the ants and bees dataset. After training, it saves the best   
   model weights as prediction_model.pth.

### Key Points
  - Fine-Tuning: Only the final classification layer is retained, adapting EfficeintNet-B0 to the specific   
  categories

--- 

## Using the Model for predictions

1. Prepare an Image: place an image in the project directory for testing
2. Run the prediction Script: python3 predict.py

--- 

## Requirements

- torch: For neural network operations
- torchvision: For pre-trained models and image transformation
- Pillow: for image loading and processing
- numpy, matplotlib: For data manipulation and visualization

--- 

## Theory (for study)

1. Pre-trained Model
   - transfer leaning begins with a model that has been trained on a large dataset. In this project, the pre- 
   trained model is used as a starting point, carrying learned patterns that can be useful for various tasks. 
2. Fine-Tuning: freeze the initial layers and only retrain the final classifier layer. It preserves the general 
   features learned on the large dataset and adapts only the output layer to our specific categories. It allows 
   the model to generalize well on our data set with relatively few training samples. (It ensures that a transfer 
   learning can be done with the small samples).
3. Transfer Learning: it reduces training time by reusing knowledge from previous tasks and requires fewer data 
  and less computational power because it often yields higher accuracy on small datasets compared to training 
  from scratch. 

   


