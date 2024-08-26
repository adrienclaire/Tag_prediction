import nltk

# Ensure that the required NLTK data files are downloaded
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

import os
import boto3
from flask import Flask, request, jsonify, render_template
import joblib
import tensorflow_hub as hub
from utils import clean_text, transform_dl_fct  # Import functions from utils.py
import pickle
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gensim


app = Flask(__name__)

# Use this code snippet in your app.
# If you need more information about configurations
# or implementing the sample code, visit the AWS docs:
# https://aws.amazon.com/developer/language/python/



from botocore.exceptions import ClientError

def get_secret():
    secret_name = "aws_secret_keys"  
    region_name = "eu-north-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        raise e

    secret = get_secret_value_response['SecretString']
    secret_dict = json.loads(secret)  # Assuming the secret is stored as a JSON string
    return secret_dict

# Retrieve AWS secrets
secrets = get_secret()
aws_access_key_id = secrets['AWS_ACCESS_KEY_ID']
aws_secret_access_key = secrets['AWS_SECRET_ACCESS_KEY']

# Configure AWS credentials using the retrieved secrets
s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

# Define the S3 bucket and model file
bucket_name = 'openclassroomprojet5'
model_file_key = 'w2v_SVC_SL_v2.pkl'

# Download the model file from S3 if it doesn't exist locally
model_path = 'w2v_SVC_SL_v2.pkl'
if not os.path.exists(model_path):
    s3.download_file(bucket_name, model_file_key, model_path)

# Load the model
model = joblib.load(model_path)

# Load the mlb file from s3
mlb_file_name = 'mlb.pkl'
# Download the file from S3 to a local file
s3.download_file(bucket_name, mlb_file_name, mlb_file_name)

# Load the multilabelbinarizer from the downloaded file
with open(mlb_file_name, 'rb') as file:
    mlb = pickle.load(file)
print("MultiLabelBinarizer loaded successfully from s3")

# Load the trained MultiLabelBinarizer
#with open('mlb.pkl', 'rb') as file:
#    mlb = pickle.load(file)
#print("MultiLabelBinarizer loaded successfully")

# Load the Universal Sentence Encoder
#print("Loading Universal Sentence Encoder...")
#embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
#print("USE model loaded successfully")

# Load the embedding model from s3
embed_model = 'word2vec_embedding_model.h5'
# Download the file from S3 to a local file
s3.download_file(bucket_name, embed_model, embed_model)

# Load the embedding model from the downloaded file
with open(embed_model, 'rb') as file:
    mlb = pickle.load(file)
print("W2v Embedding model loaded successfully from s3")

# Load the tokenizer from s3
tokenizer = 'tokenizer.pkl'
# Download the file from S3 to a local file
s3.download_file(bucket_name, tokenizer, tokenizer)

# Load the embedding model from the downloaded file
with open(tokenizer, 'rb') as file:
    tokenizer = pickle.load(file)
print("Tokenizer loaded successfully from s3")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health_check():
    return 'OK', 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['text']
    
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Tokenization and simple_preprocess
    sentence = [gensim.utils.simple_preprocess(text)]
    
    # Tokenize the input text
    sequence = pad_sequences(tokenizer.texts_to_sequences(sentence), maxlen=120, padding='post')

    # Embeded
    embeddings = embed_model.predict(sequence)
    
    # Predict tags (this will return probabilities for each tag)
    y_pred_proba = model.predict(embedding)
    
    # Log probabilities for debugging
    print("Predicted Probabilities:", y_pred_proba)
    
    # Set a threshold for prediction confidence
    threshold = 0.1  # Adjust this threshold based on your needs
    print(f"Threshold: {threshold}")
    
    # Sort the predicted probabilities in descending order
    sorted_indices = (-y_pred_proba).argsort()[0]
    top_indices = []
    
    # Select indices of tags above the threshold
    for i in sorted_indices:
        if y_pred_proba[0, i] >= threshold:
            top_indices.append(i)
        if len(top_indices) >= 3:
            break
    
    # Check the actual probabilities of the selected tags
    top_probs = y_pred_proba[0, top_indices]
    print("Top Probabilities:", top_probs)
    
    # Convert the selected indices to actual tags
    top_tags = [mlb.classes_[i] for i in top_indices]
    
    # Log selected tags for debugging
    print("Selected Tags:", top_tags)
    
    return jsonify({'predicted_tags': top_tags})

if __name__ == '__main__':
    app.run(debug=True)