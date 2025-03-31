"""
MNIST Model

This module provides model definitions for the MNIST dataset.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def create_mnist_model():
    """
    Create a simple CNN model for MNIST.
    
    Returns:
        A compiled Keras model for MNIST classification.
    """
    model = models.Sequential([
        # Input layer (MNIST images are 28x28x1)
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional layer
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third convolutional layer
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')  # 10 classes for MNIST
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model 