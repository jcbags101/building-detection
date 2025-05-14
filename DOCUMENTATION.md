# Building Detection Application Documentation

## Overview

This application is a deep learning-based system for classifying building types from images. It uses a Convolutional Neural Network (CNN) built with TensorFlow and provides a web interface using Streamlit.

## System Architecture

### 1. Model Architecture (`model.py`)

The CNN model consists of:

- Input layer: Accepts 150x150x3 RGB images
- Three convolutional blocks:
  - Block 1: 32 filters, 3x3 kernel, ReLU activation
  - Block 2: 64 filters, 3x3 kernel, ReLU activation
  - Block 3: 64 filters, 3x3 kernel, ReLU activation
- Each block includes:
  - Conv2D layer
  - MaxPooling2D layer (2x2)
- Dense layers:
  - Flatten layer
  - Dense layer with 64 units and ReLU activation
  - Output layer with softmax activation

### 2. Data Processing (`utils.py`)

#### Image Preprocessing

- Resizes images to 150x150 pixels
- Normalizes pixel values to range [0, 1]
- Converts images to RGB format
- Adds batch dimension for model input

#### Training Data Loading

- Organizes data by building categories
- Supports multiple image formats (jpg, jpeg, png)
- Handles invalid images gracefully
- Provides detailed loading statistics

### 3. Web Interface (`app.py`)

Features:

- Image upload functionality
- Real-time prediction
- Confidence score display
- Class probability distribution
- Error handling for missing model

## Setup and Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Virtual environment (recommended)

### Dependencies

```
tensorflow-cpu>=2.8.0
streamlit>=1.22.0
numpy>=1.21.0
Pillow>=9.0.0
```

### Installation Steps

1. Create virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Structure

```
dataset/
├── building_type_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── building_type_2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

## Model Training

### Training Process

1. Data Preparation:

   - Images are loaded from the dataset directory
   - Each subdirectory represents a building class
   - Images are preprocessed and normalized

2. Model Training:

   - 80-20 train-validation split
   - 10 epochs
   - Batch size of 32
   - Adam optimizer
   - Sparse categorical crossentropy loss

3. Output Files:
   - `building_classifier.h5`: Trained model
   - `class_names.txt`: Class labels

### Running Training

```bash
python model.py
```

## Web Application Usage

### Starting the App

```bash
streamlit run app.py
```

### Using the Interface

1. Upload an image using the file uploader
2. View the uploaded image
3. See prediction results:
   - Predicted building type
   - Confidence score
   - Probability distribution across all classes

## Deployment

### Local Deployment

1. Train the model
2. Start the Streamlit app
3. Access via localhost

### Cloud Deployment (Streamlit Cloud)

1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy the app

## File Structure

```
.
├── app.py              # Streamlit web interface
├── model.py            # CNN model definition and training
├── utils.py            # Utility functions
├── requirements.txt    # Project dependencies
├── dataset/            # Training data directory
├── building_classifier.h5  # Trained model
└── class_names.txt     # Class labels
```

## Error Handling

- Invalid image formats
- Missing model file
- Non-RGB images
- Empty dataset directory
- Model loading errors

## Performance Considerations

- Uses tensorflow-cpu for deployment
- Image resizing for consistent input
- Batch processing for predictions
- Efficient data loading and preprocessing

## Limitations

- Fixed input size (150x150)
- RGB images only
- Requires pre-trained model
- Limited to classes in training data

## Future Improvements

1. Support for more image formats
2. Dynamic input sizes
3. Model fine-tuning interface
4. Batch prediction support
5. API endpoint for programmatic access
