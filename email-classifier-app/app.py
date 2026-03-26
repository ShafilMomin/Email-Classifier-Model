# app.py - BERT Email Classifier
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os

app = Flask(__name__)
CORS(app)

# Load BERT model
model_path = os.path.join(os.path.dirname(__file__), 'model')
print(f"Loading BERT model from: {model_path}")

try:
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    print("✅ BERT Model loaded successfully!")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

def classify_email(email_text):
    """Classify email using BERT"""
    inputs = tokenizer(
        email_text,
        return_tensors='pt',
        truncation=True,
        max_length=128,
        padding=True
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1)
        confidence = torch.max(probabilities).item()
    
    return {
        'is_spam': prediction.item() == 1,
        'confidence': confidence,
        'spam_probability': probabilities[0][1].item(),
        'ham_probability': probabilities[0][0].item()
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    email_text = data.get('email', '')
    
    if not email_text:
        return jsonify({'error': 'No email provided'}), 400
    
    result = classify_email(email_text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)