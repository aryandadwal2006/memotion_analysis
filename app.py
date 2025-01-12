# app.py

from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from transformers import BertTokenizer
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
import os
import gdown
from transformers import TFBertModel
app = Flask(__name__)
model_url = 'https://drive.google.com/uc?id=1_QJB_SKak4wzFup7Z67YprmwBeTMPPUS'
model_path = 'models/sentiment_classification_model.h5'
def download_model():
    if not os.path.exists(model_path):
        print("Downloading model...")
        gdown.download(model_url, model_path, quiet=False)
    else:
        print("Model already exists.")
# Call the function to download the model
download_model()        
# Load the model and tokenizer
model = load_model('models/sentiment_classification_model.h5', custom_objects={'TFBertModel': TFBertModel})
tokenizer = BertTokenizer.from_pretrained('models/tokenizer/')

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define the sentiment mapping
sentiment_mapping = {0: 'very_positive', 1: 'positive', 2: 'neutral', 3: 'negative', 4: 'very_negative'}

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

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Get text and image data from the request
    text_input = data.get('text', '')
    image_data = data.get('image', '')

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

    # Make prediction
    prediction = model.predict({
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'image_input': image_input
    })

    # Map prediction to label
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = sentiment_mapping[predicted_class]
    confidence = float(np.max(prediction))

    # Prepare the response
    response = {
        'prediction': predicted_label,
        'confidence': confidence
    }

    return jsonify(response)

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)