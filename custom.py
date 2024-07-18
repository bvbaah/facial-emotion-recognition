import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, \
    Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import seaborn as sns
import cv2
import numpy as np
import gradio as gr
import json
import os

# Images in train_dir will be used to train the model
train_dir = 'dataset/train'

# Images in test_dir will be used to test the model
test_dir = 'dataset/test'

# The following commented out block of code was used to balance the dataset
# Leave commented out because the dataset is now balanced after executing code
# plus manually moving around folders because augmented sections of the dataset
# were saved within the original folders
"""
# This method will be used to augment data
# This will add pictures to folders in the dataset to balance the dataset
def augment_data(folder, num_of_images):
    p = Augmentor.Pipeline(folder, folder)

    p.rotate(probability=0.7, max_left_rotation=25, max_right_rotation=25)
    p.flip_left_right(probability=0.3)
    p.random_contrast(probability=0.3, min_factor=0.3, max_factor=1.5)
    p.random_brightness(probability=0.3, min_factor=0.5, max_factor=1.3)

    p.sample(num_of_images)

    print(f"{num_of_images} augmented images have been saved to {folder}.")


# Apply augmentation to folders to equalize the number of images in the folders
def balance_data():

    angry_dir = 'dataset/train/angry'
    augment_data(angry_dir, 3000)

    disgust_dir = 'dataset/train/disgust'
    augment_data(disgust_dir, 6500)

    fear_dir = 'dataset/train/fear'
    augment_data(fear_dir, 3000)

    # Note that happy_dir has the most amount of images so there is no need
    # to add more images
    neutral_dir = 'dataset/train/neutral'
    augment_data(neutral_dir, 2000)

    sad_dir = 'dataset/train/sad'
    augment_data(sad_dir, 2000)

    surprise_dir = 'dataset/train/surprise'
    augment_data(surprise_dir, 3800)
"""

# Create an ImageDataGenerator to randomly transform images for training
# purposes (i.e. rotate, zoom, shift)
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   # Normalize pixel values (color
                                   # description) to [0,1]
                                   rotation_range=30,
                                   # Rotate pictures up to 30 degrees
                                   shear_range=0.3,
                                   # Apply shear transformations
                                   zoom_range=0.3,  # Randomly zoom into images
                                   width_shift_range=0.3,
                                   # Randomly shift images horizontally
                                   height_shift_range=0.3,
                                   # Randomly shift images vertically
                                   horizontal_flip=True,
                                   # Randomly flip images horizontally
                                   fill_mode='nearest'
                                   # Fill in the missing pixels in images
                                   # after transformations
                                   )

# Create an ImageDataGenerator for test images (no transformations needed)
test_datagen = ImageDataGenerator(
    rescale=1. / 255)  # Normalise pixel values to [0, 1]

# Load training dataset from the directory, applying data augmentation to
# help model improve accuracy
train_generator = train_datagen.flow_from_directory(
    train_dir,  # The directory that holds the training images
    color_mode='grayscale',  # Use grayscale images
    target_size=(48, 48),  # Resize all images to 48 x 48 pixels for uniformity
    batch_size=64,  # Number of images to yield per batch
    class_mode='categorical',  # Use categorical labels for dataset
    shuffle=True  # Shuffle the data
)

# Load test dataset from the directory, without applying augmentation
test_generator = test_datagen.flow_from_directory(
    test_dir,  # The directory that holds the training images
    color_mode='grayscale',  # Use grayscale images
    target_size=(48, 48),  # Resize all images to 48 x 48 pixels for uniformity
    batch_size=64,  # Number of images to yield per batch
    class_mode='categorical',  # Use categorical labels for dataset
    shuffle=True  # Shuffle the data
)

# Build the model
# Define the CNN (convolutional neural network) algorithm
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    # First convolutional layer with 32 filters
    MaxPooling2D(2, 2),  # First max pooling layer
    Dropout(0.1),  # Dropout layer to prevent overfitting (drop 25% of units)

    Conv2D(64, (3, 3), activation='relu'),
    # Second convolutional layer with 64 filters
    MaxPooling2D(2, 2),  # Second max pooling layer
    Dropout(0.1),  # Dropout layer to prevent overfitting (drop 25% of units)

    Conv2D(128, (3, 3), activation='relu'),
    # Third convolutional layer with 128 filters
    MaxPooling2D(2, 2),  # Third max pooling layer
    Flatten(),  # Flatten the 3D output to 1D
    Dense(128, activation='relu'),  # This is a fully connected layer wth 128
    Dropout(0.2),  # Dropout layer to prevent overfitting (drop 50% of units)
    Dense(7, activation='softmax')
    # Output layer with 7 units - one for each emotion
])

# Compile the model
model.compile(optimizer="adam",
              loss=['categorical_crossentropy'],
              metrics=['accuracy'])

# Callbacks
# Add early stopping to prevent overfitting
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

# Print the model summary to see how the CNN architecture functions
model.summary()


# Train the model on the training data and test the data on the test data
def train_and_save_model():
    # Train the model
    model_info = model.fit(train_generator,
                           validation_data=test_generator,
                           epochs=30)

    # Save the model
    model.save('custom_model.h5')

    # Save training history
    with open('custom_model_info.json', 'w') as f:
        json.dump(model_info.history, f)

    return model_info


def show_graphs():
    # Load the trained model
    saved_model = load_model('custom_model.h5')

    # Load training history
    with open('custom_model_info.json', 'r') as f:
        custom_model_info = json.load(f)

    # Extract and print accuracy per epoch
    accuracy = custom_model_info['accuracy']
    val_accuracy = custom_model_info['val_accuracy']

    print("Training accuracy per epoch:", accuracy)
    print("Validation accuracy per epoch:", val_accuracy)

    # Find the best epoch based on validation accuracy
    val_accuracy = custom_model_info['val_accuracy']
    best_epoch = np.argmax(val_accuracy)
    best_val_accuracy = val_accuracy[best_epoch]

    print(f"Best Epoch: {best_epoch + 1}")  # Adding 1 for user-friendly output
    print(f"Validation Accuracy at Best Epoch: {best_val_accuracy * 100:.2f}%")

    # Find the greatest training accuracy
    accuracy = custom_model_info['accuracy']
    greatest_train_accuracy = max(accuracy)

    print(
        f"Greatest Training Accuracy during Training: {greatest_train_accuracy * 100:.2f}%")

    # Evaluate the model on the test data
    #test_loss, test_accuracy = model.evaluate(test_generator)
    #print(f"Overall test accuracy: {test_accuracy * 100:.2f}%")

    # Get the test data and labels
    test_generator.reset()
    predictions = saved_model.predict(test_generator,
                                      steps=test_generator.samples // test_generator.batch_size + 1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    # Class labels
    class_labels = list(test_generator.class_indices.keys())

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    conf_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                          display_labels=class_labels)
    conf_display.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

    # Accuracy Line Graph
    plt.figure()
    plt.plot(custom_model_info['accuracy'], label='Training Accuracy')
    plt.plot(custom_model_info['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.show()

    # Bar Graph for Confidence Levels
    # Select a batch of images from the test set
    test_generator.reset()
    x_batch, y_batch = next(test_generator)

    # Predict the probabilities
    probs = saved_model.predict(x_batch)

    # Get mean confidence levels for each emotion
    mean_confidence = np.mean(probs, axis=0)

    # Bar plot
    plt.figure()
    plt.bar(class_labels, mean_confidence, color='skyblue')
    plt.title('Mean Confidence Levels for Each Emotion')
    plt.ylabel('Confidence')
    plt.xlabel('Emotion')
    plt.show()


if __name__ == "__main__":

    # The following line will train and save model
    # This line is left commented such that the code will use the
    # saved model to generate graphs instead of always training and saving model
    # If model needs to be retrained, uncomment the following line
    #train_and_save_model()
    # Keep this line uncommented to show graphs
    #show_graphs()
