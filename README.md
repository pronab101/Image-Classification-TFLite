# Image-Classification-TFLite

An image classification project using TensorFlow and Keras to create a CNN model, train it on a custom dataset, and deploy it in TensorFlow Lite (TFLite) format for compatibility with mobile and edge devices.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Model Conversion to TFLite](#model-conversion-to-tflite)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project is designed for **image classification** using a Convolutional Neural Network (CNN). The model is trained on a dataset of categorized images and is optimized for deployment in TensorFlow Lite format, making it lightweight for mobile and embedded device usage.

## Features
- **Data Preprocessing**: Uses ImageDataGenerator to rescale and augment images.
- **Custom CNN Model**: Built with TensorFlow and Keras for multi-class classification.
- **TFLite Conversion**: Converts the trained model to TensorFlow Lite format for efficient deployment on mobile and edge devices.
  
## Dataset
The dataset should be structured as follows:

dataset/
│
├── train/
│   ├── class1/
│   ├── class2/
│   └── class3/
│
└── validation/
    ├── class1/
    ├── class2/
    └── class3/

Each class directory contains images belonging to that category.

## Model Architecture
The CNN model is defined with three convolutional layers, followed by max-pooling, a dense layer with ReLU activation, and an output layer with softmax activation for multi-class classification.

## Installation
1. Clone the repository:
    
    git clone https://github.com/pronab101/Image-Classification-TFLite.git
    cd Image-Classification-TFLite
    
2. Install the required dependencies:
   
    pip install tensorflow
   

## Usage
1. **Update Dataset Paths**: Ensure `train_dir` and `val_dir` are set to your dataset paths in the script.
2. **Run the Model**: Use the following command to start training and save the model:
   
    python train_model.py
   
3. **Evaluate the Model**: After training, the script will output accuracy metrics for both training and validation datasets.

## Model Conversion to TFLite
After training, the model is saved in `.h5` format and then converted to TFLite:

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()


To save the TFLite model:

with open("model.tflite", "wb") as f:
    f.write(tflite_model)


## Results
The trained model achieved an Validation Accuracy 100.00% on the validation set (adjust this based on your actual results). You may add sample predictions or a confusion matrix here for additional context.

## Contributing
Contributions are welcome! If you'd like to improve this project, please fork the repository and create a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

This README template is comprehensive yet concise, providing clear information about the project structure, installation, usage, and additional details to help others understand and use your project.
