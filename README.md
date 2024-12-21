# MNIST Classification with PyTorch

A simple PyTorch implementation achieving 99.4% accuracy on MNIST using less than 20k parameters.

## Features
- Batch Normalization
- Dropout
- Data Augmentation
- OneCycleLR scheduler
- Less than 20k parameters

## Project Structure
MNIST-train/
├── model.py # Main model architecture and training code
├── requirements.txt # Project dependencies
└── .github/
└── workflows/ # CI/CD pipelines

## Model Architecture
- CNN with batch normalization and dropout
- Input: 28x28 grayscale images
- Output: 10 classes (digits 0-9)
- Parameters: < 20k
- Target accuracy: 99.4% on test set
- Uses batch normalization after each convolution layer
- Includes dropout for regularization
- Uses Global Average Pooling to reduce parameters
- Has less than 20k parameters
- Uses a learning rate scheduler to improve convergence
- Includes proper metrics tracking
- Saves the best model based on accuracy
- Has early stopping when target accuracy is reached
- Data augmentation to improve generalization

## Model learning target
- 99.4% accuracy in test set    
- Less than 20k parameters


## Requirements
- torch>=1.7.0
- torchvision>=0.8.0
- numpy>=1.19.2
- tqdm>=4.50.0
- torchsummary>=1.5.1

## Usage
- Clone the repository
- Install the dependencies
- Run the model

## Model Features
1. **Data Augmentation**
   - Random rotation (-7° to 7°)
   - Random affine translations

2. **Regularization**
   - Batch Normalization after convolutions
   - Dropout (5%)
   - Weight decay: 5e-4

3. **Training**
   - OneCycleLR scheduler
   - SGD optimizer with momentum
   - Batch size: 128

## Results
- Test Accuracy: 99.4%
- Training Time: < 20 epochs
- Parameter Count: < 20k

## CI/CD
GitHub Actions workflows verify:
- Model architecture requirements
- Parameter count limits
- Presence of BatchNorm and Dropout
- Model forward pass functionality

## Test validation
![image](https://github.com/user-attachments/assets/cb2a2c9b-9279-4775-987f-d7b31f7b45e5)
