from flask import Flask, render_template, request, jsonify  # Import necessary modules from Flask
import base64  # Module to handle base64 encoding for images
from openai import OpenAI  # Import OpenAI library to interact with GPT-4 API
import openai
from dotenv import load_dotenv
import os
# Initialize the Flask app
app = Flask(__name__)

# Your OpenAI API Key here (Replace with your actual API Key)
client = OpenAI()  # Instantiate OpenAI client
API_KEY = os.getenv('APIKEY')

# Function to encode an uploaded image as a base64 string
def encode_image(image):
    # Read the image and encode it in base64 format, then decode to a UTF-8 string
    return base64.b64encode(image.read()).decode('utf-8')

# Route to render the home page
@app.route('/')
def home():
    # Render 'index.html' as the homepage
    return render_template('index.html')

# Route to handle image classification requests
@app.route('/classify', methods=['POST'])
def classify_image():
    # Check if 'image' file is part of the request; if not, return error response
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400  # 400 status for client error

    image = request.files['image']  # Get the uploaded image file from the request
    # Check if the uploaded file has a name (i.e., was a file selected?)
    if not image.filename:
        return jsonify({'error': 'No image selected.'}), 400  # Error if no file selected

    try:
        # Encode the image in base64 format for sending to GPT-4 Vision API
        base64_image = encode_image(image)

        # Make an API call to GPT-4 Vision using the OpenAI client
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Model to use for image classification
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What’s in this image?"},  # User prompt text
                        {
                            "type": "image_url",  # Specify that an image follows in the message
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"  # Send image as base64 data URL
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,  # Limit the response to 300 tokens
        )
        
        # Extract the model's classification result from the response
        result = response.choices[0].message.content
        # Return the classification result to the frontend in JSON format
        return jsonify({"classification": result})

    except Exception as e:
        # Handle any exceptions by returning a descriptive error message and a 500 status code
        return jsonify({'error': f'Classification failed: {str(e)}'}), 500

# Main entry point for running the app
if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app in debug mode (useful for development)
