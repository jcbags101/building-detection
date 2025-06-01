# Building Type Classification

This project implements a CNN-based model for classifying different types of buildings from images. The model is built using TensorFlow/Keras and includes a Streamlit web interface for easy interaction.

## Project Structure

```
building-detection/
├── app.py              # Streamlit web application
├── model.py            # CNN model implementation and training
├── utils.py            # Utility functions for data loading and preprocessing
├── requirements.txt    # Python dependencies
├── dataset/           # Training data directory
│   ├── class1/        # Images for class 1
│   ├── class2/        # Images for class 2
│   └── ...
└── models/            # Directory for saved models
```

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Organization

The dataset should be organized in the following structure:

```
dataset/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

Each class should be in its own directory, containing the respective building images. Supported image formats are: .jpg, .jpeg, and .png.

## Training the Model

To train the model, run:

```bash
python model.py
```

The training process will:

1. Load and preprocess images from the dataset directory
2. Split the data into training (64%), validation (16%), and test (20%) sets
3. Train a CNN model with the following architecture:
   - 3 convolutional layers with max pooling
   - Dense layers for classification
4. Generate and save:
   - The trained model (`building_classifier.h5`)
   - Class names (`class_names.txt`)
   - Confusion matrix visualization (`confusion_matrix.png`)
   - Training history plots (`training_history.png`)

## Model Architecture

The CNN model consists of:

- Input layer: 150x150x3 (RGB images)
- 3 convolutional blocks, each with:
  - Conv2D layer with ReLU activation
  - MaxPooling2D layer
- Flatten layer
- Dense layer (64 units) with ReLU activation
- Output layer with softmax activation

## Running the Web Interface

To start the Streamlit web interface:

```bash
streamlit run app.py
```

The web interface allows you to:

1. Upload building images
2. Get predictions for the building type
3. View the model's confidence scores

## Model Evaluation

The training process automatically generates:

- Confusion matrix showing prediction accuracy across classes
- Classification report with precision, recall, and F1-score
- Training history plots showing accuracy and loss over epochs

## Requirements

- Python 3.7+
- TensorFlow 2.8.0+
- Streamlit 1.22.0+
- NumPy 1.21.0+
- Pillow 9.0.0+
- scikit-learn 1.0.0+
- matplotlib 3.5.0+
- seaborn 0.11.0+
