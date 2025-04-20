import os
import numpy as np
from PIL import Image
import tensorflow as tf

def preprocess_image(image_path, target_size=(150, 150)):
    """
    Preprocess a single image for prediction.
    
    Args:
        image_path: Path to the image file
        target_size: Tuple of (height, width) to resize the image to
        
    Returns:
        Preprocessed image as numpy array
    """
    # Load and resize image
    img = Image.open(image_path)
    img = img.resize(target_size)
    
    # Convert to array and normalize
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def load_training_data(data_dir, target_size=(150, 150)):
    """
    Load and preprocess training data from directory.
    
    Args:
        data_dir: Path to the dataset directory
        target_size: Tuple of (height, width) to resize images to
        
    Returns:
        Tuple of (images, labels, class_names)
    """
    images = []
    labels = []
    class_names = []
    
    # Get all subdirectories (each represents a class)
    subdirs = [d for d in sorted(os.listdir(data_dir)) 
              if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]
    
    # First, collect class names
    class_names = subdirs
    
    # Then load images with proper label indices
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        print(f"Loading class {class_name} (index {class_idx})")
        
        # Load all images in this class
        for image_name in os.listdir(class_dir):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(class_dir, image_name)
                
                # Load and preprocess image
                try:
                    img = Image.open(image_path)
                    img = img.resize(target_size)
                    img_array = np.array(img)
                    
                    # Skip images that don't have 3 channels
                    if len(img_array.shape) != 3 or img_array.shape[2] != 3:
                        print(f"Skipping {image_path} - not a valid RGB image")
                        continue
                        
                    img_array = img_array.astype('float32') / 255.0
                    
                    images.append(img_array)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading {image_path}: {str(e)}")
                    continue
    
    if not images:
        raise ValueError("No valid images found in the dataset directory")
    
    print(f"Loaded {len(images)} images across {len(class_names)} classes")
    print(f"Class names: {class_names}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    return np.array(images), np.array(labels), class_names

def load_model(model_path):
    """
    Load the trained model from file.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded Keras model
    """
    return tf.keras.models.load_model(model_path) 