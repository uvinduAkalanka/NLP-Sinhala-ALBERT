from flask import Flask, request, jsonify
import torch ,os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

app = Flask(__name__)

# Load the model and tokenizer
def get_classifier():
    model_name = 'models/sinhala_albert_100_epochs'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define the model
    id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3, 
        id2label=id2label, 
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.max_position_embeddings = tokenizer.model_max_length

    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return classifier

# Initialize the classifier
classifier = get_classifier()

# Define a route for the sentiment analysis API
@app.route('/analyze', methods=['GET'])
def analyze():
    try:
        # Get the text input from the query parameters
        text = request.args.get('text')
        print(f"Received text: {text}")  # Debug print statement
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Perform sentiment analysis
        result = classifier(text)
        print(f"Analysis result: {result}")  # Debug print statement
        
        # Return the result as JSON
        return jsonify(result)
    except Exception as e:
        print(f"Error: {str(e)}")  # Debug print statement
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)