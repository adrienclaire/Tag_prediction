import os
import boto3
from flask import Flask, request, jsonify
import joblib
import tensorflow_hub as hub
from utils import clean_text, transform_dl_fct  # Import functions from utils.py
import pickle
import json


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
model_file_key = 'USE_SVC_SL.pkl'

# Download the model file from S3 if it doesn't exist locally
model_path = 'USE_SVC_SL.pkl'
if not os.path.exists(model_path):
    s3.download_file(bucket_name, model_file_key, model_path)

# Load the model
model = joblib.load(model_path)

# Load the trained MultiLabelBinarizer
with open('mlb.pkl', 'rb') as file:
    mlb = pickle.load(file)
print("MultiLabelBinarizer loaded successfully")

# Load the Universal Sentence Encoder
print("Loading Universal Sentence Encoder...")
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
print("USE model loaded successfully")

@app.route('/health')
def health_check():
    return 'OK', 200

@app.route('/')
def home():
    return 'OK', 200


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['text']
    
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Tokenize the cleaned text
    tokenized_text = transform_dl_fct(cleaned_text)
    
    # Convert to embedding
    embedding = embed([tokenized_text])
    
    # Predict tags
    y_pred_binary = model.predict(embedding)
    
    # Convert binary predictions to actual tags
    predicted_tags = mlb.inverse_transform(y_pred_binary)
    
    return jsonify({'predicted_tags': predicted_tags[0]})

if __name__ == '__main__':
    app.run()
