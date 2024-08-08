import os
import boto3
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Configure AWS credentials from environment variables
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

# Define the S3 bucket and model file
bucket_name = 'your-bucket-name'
model_file_key = 'model.pkl'

# Download the model file from S3 if it doesn't exist locally
model_path = 'model.pkl'
if not os.path.exists(model_path):
    s3.download_file(bucket_name, model_file_key, model_path)

# Load the model
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['text']
    # Preprocess the text here (e.g., tokenization, padding)
    # Assuming the text is preprocessed into 'padded_sequence'
    
    # Predict tags
    prediction = model.predict([padded_sequence])
    
    # Convert prediction to tags
    predicted_tags = list(prediction[0])
    
    return jsonify({'predicted_tags': predicted_tags})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
