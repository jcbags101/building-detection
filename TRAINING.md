# Report on Training a Building Type Classifier

**Course:** [Your Course Name/Number Here]
**Date:** October 26, 2023
**Student Name:** [Your Name Here]

## 1. Introduction

This report details the development and training of a Convolutional Neural Network (CNN) for classifying building types from images. The primary goal of this project was to create a functional application capable of identifying different building categories (e.g., 'church', 'hospital') based on visual input. The project involved data preparation, model design, training, and the creation of a user-friendly web interface for predictions.

## 2. Dataset Preparation

### 2.1. Data Source and Structure

The image data for this project was sourced from [Mention your data source, e.g., "a custom-collected dataset of building images," or "a publicly available dataset like ..."]. The dataset was organized into a hierarchical folder structure:

- A root `dataset/` folder.
- Subfolders within `dataset/`, where each subfolder name corresponds to a specific building category (e.g., `dataset/church/`, `dataset/hospital/`).
- Image files (e.g., `.jpg`, `.png`, `.jpeg`) were placed within their respective category subfolders.

This structure allowed for straightforward loading and automatic labeling of images based on their parent directory.

### 2.2. Image Preprocessing

Before being fed into the neural network, the images underwent several preprocessing steps, handled by functions in `utils.py`:

1.  **Loading**: Images were loaded using the Pillow (PIL) library.
2.  **Resizing**: All images were resized to a uniform dimension of 150x150 pixels. This standardization is crucial for CNNs, which expect fixed-size inputs.
3.  **Normalization**: Pixel values, typically ranging from 0 to 255, were scaled to a range of 0 to 1 by dividing each pixel value by 255.0. Normalization helps stabilize and speed up the training process.
4.  **Channel Consistency**: The code ensures that images are treated as 3-channel (RGB) images. Grayscale or images with alpha channels that don't conform are skipped to maintain consistency.
5.  **Batch Dimension**: For individual image predictions (in the Streamlit app), an extra batch dimension was added to the image array to match the input shape expected by the trained model.

## 3. Model Architecture and Algorithm

A Convolutional Neural Network (CNN) was chosen as the core algorithm for this image classification task. CNNs are well-suited for visual data due to their ability to automatically and adaptively learn spatial hierarchies of features from input images.

The specific architecture, defined in `model.py`, is a sequential model built using TensorFlow's Keras API:

1.  **Input Layer**: Implicitly defined by the `input_shape=(150, 150, 3)` in the first `Conv2D` layer, indicating 150x150 pixel images with 3 color channels (RGB).
2.  **Convolutional Block 1**:
    - `Conv2D` layer: 32 filters, kernel size of (3,3), ReLU (Rectified Linear Unit) activation function.
    - `MaxPooling2D` layer: Pool size of (2,2). This layer reduces the spatial dimensions (height and width) of the feature maps.
3.  **Convolutional Block 2**:
    - `Conv2D` layer: 64 filters, kernel size of (3,3), ReLU activation.
    - `MaxPooling2D` layer: Pool size of (2,2).
4.  **Convolutional Block 3**:
    - `Conv2D` layer: 64 filters, kernel size of (3,3), ReLU activation.
    - `MaxPooling2D` layer: Pool size of (2,2).
5.  **Flatten Layer**: This layer converts the 2D feature maps from the convolutional blocks into a 1D feature vector, which can be fed into dense layers.
6.  **Dense Layer 1**: A fully connected layer with 64 units and ReLU activation. This layer learns non-linear combinations of the features extracted by the convolutional layers.
7.  **Output Dense Layer**: A fully connected layer with `num_classes` units (where `num_classes` is the number of building categories detected in the dataset) and a `softmax` activation function. The softmax function outputs a probability distribution over the classes, indicating the model's confidence for each building type.

**Rationale for CNNs:**
CNNs excel at image-based tasks because:

- **Local Receptive Fields**: Neurons in early layers connect to small regions of the input, capturing local patterns like edges and textures.
- **Shared Weights**: The same filter (set of weights) is applied across different parts of the image, allowing the model to detect a feature regardless of its position. This significantly reduces the number of parameters.
- **Hierarchical Feature Learning**: Subsequent layers learn more complex features by combining features from previous layers.

## 4. Training Procedure

The model training was orchestrated by the `train_model` function in `model.py`.

### 4.1. Data Loading and Splitting

1.  **Data Loading**: The `load_training_data` function from `utils.py` was used to load all images from the `dataset/` directory, preprocess them, and assign numerical labels based on their subfolder.
2.  **Train-Validation Split**: The loaded dataset (images and labels) was split into training and validation sets using `sklearn.model_selection.train_test_split`. An 80% - 20% split was used (test_size=0.2), with `random_state=42` to ensure reproducibility of the split.
    - The **training set** is used to teach the model.
    - The **validation set** is used to evaluate the model's performance on unseen data during training, helping to monitor for overfitting.

### 4.2. Model Compilation

Before training, the model was compiled with the following configurations:

- **Optimizer**: `adam` (Adaptive Moment Estimation). Adam is an efficient and commonly used optimization algorithm that adapts the learning rate for each parameter.
- **Loss Function**: `sparse_categorical_crossentropy`. This loss function is suitable for multi-class classification problems where the labels are integers (e.g., 0, 1, 2...).
- **Metrics**: `accuracy`. This metric was monitored during training to evaluate the percentage of images correctly classified.

### 4.3. Model Fitting (Training)

The model was trained using the `fit` method:

- **Epochs**: 10. An epoch is one complete pass through the entire training dataset.
- **Batch Size**: 32. The training data was fed to the model in batches of 32 images at a time.
- **Validation Data**: The `(X_val, y_val)` set was provided to evaluate the model's performance on the validation set after each epoch.

### 4.4. Model Saving

After training completed:

1.  The trained model (weights and architecture) was saved to a file named `building_classifier.h5` using the HDF5 format.
2.  The list of class names (building categories) was saved to `class_names.txt`. This file is used by the Streamlit application to map the model's numerical output back to human-readable class names.

## 5. Results and Evaluation (During Training)

The training process included monitoring the accuracy and loss on both the training and validation datasets for each epoch. This allows for an assessment of how well the model is learning and generalizing. For this project, the primary outcome was a saved `building_classifier.h5` model file, ready for use in the prediction application.

_(Note: For a complete school report, you would typically include graphs of training/validation loss and accuracy over epochs, and potentially an evaluation on a separate test set with metrics like precision, recall, F1-score, and a confusion matrix. The current script focuses on training and saving the model for the Streamlit app.)_

## 6. Tools and Libraries Used

- **Python**: The core programming language.
- **TensorFlow with Keras API**: For defining, training, and saving the Convolutional Neural Network.
- **Pillow (PIL)**: For image loading and manipulation (resizing).
- **NumPy**: For numerical operations, especially array handling.
- **scikit-learn**: For splitting the dataset into training and validation sets.
- **Streamlit**: For creating the interactive web application to use the trained model.

## 7. Conclusion

This project successfully demonstrated the process of building an image classification system using a Convolutional Neural Network. The steps involved dataset preparation, model design using a standard CNN architecture, and a robust training procedure. The resulting trained model can now be used via the Streamlit web interface to classify building types from user-uploaded images. Further work could involve expanding the dataset, experimenting with more complex architectures, and implementing more detailed performance evaluation.
