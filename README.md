# Building Type Classifier

This application uses a Convolutional Neural Network (CNN) to classify building types from images. It includes a web interface built with Streamlit for easy interaction.

## Local Setup

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Prepare your dataset:
   - Create a `dataset` folder in the project root
   - Inside `dataset`, create subfolders for each building type (e.g., 'house', 'office', 'church')
   - Place corresponding images in each subfolder

## Training the Model

To train the model on your dataset:

```bash
python model.py
```

This will:

- Load and preprocess images from the `dataset` folder
- Train a CNN model
- Save the trained model as `building_classifier.h5`
- Save class names to `class_names.txt`

## Running the Web App

To start the Streamlit web interface locally:

```bash
streamlit run app.py
```

## Deployment

This app can be deployed to Streamlit Cloud for free. To deploy:

1. Create a GitHub repository and push your code:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository, branch (main), and file (app.py)
6. Click "Deploy"

Note: Make sure your trained model (`building_classifier.h5`) and class names file (`class_names.txt`) are committed to the repository.

## Project Structure

- `model.py`: Contains the CNN model definition and training code
- `utils.py`: Helper functions for image preprocessing and model loading
- `app.py`: Streamlit web interface
- `requirements.txt`: Project dependencies
- `dataset/`: Directory for training images (create this)
- `building_classifier.h5`: Saved model (created after training)
- `class_names.txt`: List of class names (created after training)
