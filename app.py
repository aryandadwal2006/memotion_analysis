# app.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING messages

from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from transformers import BertTokenizer, TFBertModel
import numpy as np
import tensorflow as tf
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import contractions
import base64
from PIL import Image
import io
import gdown
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

model_path = 'models/sentiment_classification_model.h5'

def download_model():
    if not os.path.exists(model_path):
        logging.info("Downloading model...")
        try:
            # Use the file ID directly
            file_id = '1_QJB_SKak4wzFup7Z67YprmwBeTMPPUS'
            gdown.download(id=file_id, output=model_path, quiet=False)
            logging.info("Model downloaded successfully.")
        except Exception as e:
            logging.error(f"Error downloading the model: {e}")
            exit(1)  # Exit if download fails
    else:
        logging.info("Model already exists.")

# Call the function to download the model
download_model()

# Load the model and tokenizer
try:
    model = load_model(model_path, custom_objects={'TFBertModel': TFBertModel})
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading the model: {e}")
    exit(1)

# Option A: Load tokenizer from local files
# tokenizer = BertTokenizer.from_pretrained('models/tokenizer/')

# Option B: Load pretrained tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Ensure NLTK data is downloaded
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)

# Define the sentiment mapping
sentiment_mapping = {
    0: 'very_positive',
    1: 'positive',
    2: 'neutral',
    3: 'negative',
    4: 'very_negative'
}

# Define the text preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    clean_text = ' '.join(tokens)
    return clean_text

# Define the image preprocessing function
def preprocess_image(image_data):
    try:
        # Decode base64 image data
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        # Resize the image
        image = image.resize((224, 224))
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Convert image to numpy array
        img_array = np.array(image)
        # Expand dimensions to match model's input_shape
        img_array = np.expand_dims(img_array, axis=0)
        # Preprocess image data
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        return img_array
    except Exception as e:
        logging.error(f"Error preprocessing image: {e}")
        return None

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Get text and image data from the request
        text_input = data.get('text', '')
        image_data = data.get('image', '')

        if not image_data:
            return jsonify({'error': 'No image data provided.'}), 400

        # Preprocess text
        processed_text = preprocess_text(text_input)
        encoded = tokenizer.encode_plus(
            processed_text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False
        )
        input_ids = np.array([encoded['input_ids']])
        attention_masks = np.array([encoded['attention_mask']])

        # Preprocess image
        image_input = preprocess_image(image_data)
        if image_input is None:
            return jsonify({'error': 'Error processing image data.'}), 400

        # Make prediction
        prediction = model.predict({
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'image_input': image_input
        })

        # Map prediction to label
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = sentiment_mapping.get(predicted_class, 'unknown')
        confidence = float(np.max(prediction))

        # Prepare the response
        response = {
            'prediction': predicted_label,
            'confidence': confidence
        }

        return jsonify(response)

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
