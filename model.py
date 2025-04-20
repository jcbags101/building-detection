import os
import tensorflow as tf
from tensorflow.keras import layers, models
from utils import load_training_data

def create_model(num_classes):
    """
    Create a simple CNN model for building classification.
    
    Args:
        num_classes: Number of building classes to predict
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # First convolutional layer
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional layer
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third convolutional layer
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(data_dir='dataset', model_path='building_classifier.h5'):
    """
    Train the model on the dataset and save it.
    
    Args:
        data_dir: Path to the dataset directory
        model_path: Path where to save the trained model
    """
    # Load and preprocess the data
    print("Loading training data...")
    images, labels, class_names = load_training_data(data_dir)
    
    # Print dataset information
    print(f"Found {len(class_names)} classes: {class_names}")
    print(f"Total images: {len(images)}")
    
    # Split into training and validation sets
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )
    
    # Create and train the model
    print("Creating and training model...")
    model = create_model(len(class_names))
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_val, y_val)
    )
    
    # Save the model
    print(f"Saving model to {model_path}...")
    model.save(model_path)
    
    # Save class names
    with open('class_names.txt', 'w') as f:
        f.write('\n'.join(class_names))
    
    print("Training complete!")
    return model, history

if __name__ == '__main__':
    train_model() 