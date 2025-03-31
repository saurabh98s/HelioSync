#!/usr/bin/env python3
"""
Federated Learning - TensorFlow Models

This module defines TensorFlow models for different datasets.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Dict, List, Tuple, Optional, Union, Callable

def create_model(dataset_name: str) -> tf.keras.Model:
    """Create a TensorFlow model for the specified dataset.
    
    Args:
        dataset_name: Name of the dataset ('mnist', 'cifar10', etc.).
        
    Returns:
        A TensorFlow model.
    """
    if dataset_name.lower() == "mnist":
        return create_mnist_model()
    elif dataset_name.lower() == "cifar10":
        return create_cifar10_model()
    elif dataset_name.lower() == "sentiment":
        return create_sentiment_model()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def create_mnist_model() -> tf.keras.Model:
    """Create a CNN model for MNIST classification.
    
    Returns:
        A compiled Keras model.
    """
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    # Compile model with Adam optimizer
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_cifar10_model() -> tf.keras.Model:
    """Create a TensorFlow model for the CIFAR-10 dataset.
    
    Returns:
        A TensorFlow model for CIFAR-10.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10)
    ])
    
    return model

def create_sentiment_model() -> tf.keras.Model:
    """Create a TensorFlow model for sentiment analysis.
    
    Returns:
        A TensorFlow model for sentiment analysis.
    """
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(100,)),  # Assuming 100-dim embeddings
        layers.Dropout(0.5),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

def create_custom_model(
    input_shape: Tuple[int, ...],
    num_classes: int,
    hidden_layers: List[int] = [128, 64],
) -> tf.keras.Model:
    """Create a custom TensorFlow model with the specified architecture.
    
    Args:
        input_shape: Shape of the input data.
        num_classes: Number of output classes.
        hidden_layers: List of hidden layer sizes.
        
    Returns:
        A custom TensorFlow model.
    """
    model = models.Sequential()
    
    # Flatten input if needed
    if len(input_shape) > 1:
        model.add(layers.Flatten(input_shape=input_shape))
    else:
        model.add(layers.Input(shape=input_shape))
    
    # Add hidden layers
    for units in hidden_layers:
        model.add(layers.Dense(units, activation='relu'))
        model.add(layers.Dropout(0.2))
    
    # Add output layer
    if num_classes == 1:
        model.add(layers.Dense(1, activation='sigmoid'))
    elif num_classes == 2:
        model.add(layers.Dense(1, activation='sigmoid'))
    else:
        model.add(layers.Dense(num_classes))
    
    return model 