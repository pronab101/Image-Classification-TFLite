# Import TensorFlow library as tf for building and training models
import tensorflow as tf

# Import ImageDataGenerator for data augmentation and preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import os library for interacting with the operating system, such as reading directories
import os

# Paths to dataset directories and where to save the TFLite model and labels file
train_dir = '/Users/pronabkarmaker/Project/dataset/train'  # Training dataset path
val_dir = '/Users/pronabkarmaker/Project/dataset/validation'  # Validation dataset path
model_save_path = '/Users/pronabkarmaker/Project/my_model/model.tflite'  # Path to save TFLite model
label_file_path = '/Users/pronabkarmaker/Project/my_model/labels.txt'  # Path to save label file

# Parameters for data and model
image_size = (224, 224)  # Image size to which all images will be resized
batch_size = 32  # Number of samples per gradient update
class_names = [folder for folder in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, folder))]
num_classes = len(class_names) # Number of classes based on folders in the training directory
print(f"Detected classes: {class_names}")
print(f"Number of classes: {num_classes}")


# Set up data augmentation and preprocessing for training and validation sets
train_datagen = ImageDataGenerator(rescale=1./255)  # Rescale pixel values between 0 and 1 for training data
val_datagen = ImageDataGenerator(rescale=1./255)  # Rescale pixel values for validation data

# Create the training data generator from the directory
train_generator = train_datagen.flow_from_directory(
    train_dir,  # Path to training directory
    target_size=image_size,  # Resize images to the specified image size
    batch_size=batch_size,  # Number of images to yield per batch
    class_mode='categorical'  # Use categorical labels for multi-class classification
)

# Create the validation data generator from the directory
validation_generator = val_datagen.flow_from_directory(
    val_dir,  # Path to validation directory
    target_size=image_size,  # Resize images to the specified image size
    batch_size=batch_size,  # Number of images per batch
    class_mode='categorical'  # Categorical labels for validation
)



# Define a simple CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),  # First convolutional layer
    tf.keras.layers.MaxPooling2D(2, 2),  # First max pooling layer
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  # Second convolutional layer
    tf.keras.layers.MaxPooling2D(2, 2),  # Second max pooling layer
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),  # Third convolutional layer
    tf.keras.layers.MaxPooling2D(2, 2),  # Third max pooling layer
    tf.keras.layers.Flatten(),  # Flatten layer to convert 3D output to 1D
    tf.keras.layers.Dense(512, activation='relu'),  # Dense (fully connected) layer with 512 neurons
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Output layer with softmax activation for classification
])

# Compile the model with loss function, optimizer, and metric
model.compile(
    loss='categorical_crossentropy',  # Loss function for multi-class classification
    optimizer='adam',  # Adam optimizer
    metrics=['accuracy']  # Metric to evaluate the model
)

# Train the model
model.fit(
    train_generator,  # Training data generator
    epochs=10,  # Number of epochs
    validation_data=validation_generator  # Validation data generator
)

# Evaluate the model on the validation set to get final accuracy
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")  # Print validation accuracy in percentage

# Save the model in HDF5 format
h5_model_path = '/Users/pronabkarmaker/MSC Project/my_model/keras.h5'  # Path to save HDF5 model
model.save(h5_model_path)  # Save the trained model in HDF5 format

# Convert the trained model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(tf.keras.models.load_model(h5_model_path))  # Load and convert the HDF5 model
tflite_model = converter.convert()  # Convert the model to TFLite

# Save the converted TFLite model to file
with open(model_save_path, 'wb') as f:
    f.write(tflite_model)  # Write the TFLite model to the specified file
print(f"TFLite model saved as {model_save_path}")  # Print confirmation

# Create a labels file with class names
class_indices = train_generator.class_indices  # Get class indices from the training data generator
labels = list(class_indices.keys())  # List of class labels (folder names)

# Write the labels to the specified file
with open(label_file_path, 'w') as f:
    for label in labels:
        f.write(f"{label}\n")  # Write each label on a new line
print(f"Label file saved as {label_file_path}")  # Print confirmation of label file creation
