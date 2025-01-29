# Potato Leaf Detection

This project is a machine learning-based solution for detecting potato leaf diseases. The model has been deployed using FastAPI, providing an API endpoint for users to upload images of potato leaves and receive disease predictions.

## Project Overview

This project involves building and deploying a potato leaf disease detection system using a deep learning model. The model was trained using TensorFlow and leverages the Python Imaging Library (PIL) and OpenCV for image processing. The FastAPI application serves as the backend, allowing users to interact with the model via HTTP requests.

## Model Details

- **TensorFlow**: Used to build and train the deep learning model for detecting diseases in potato leaves.
- **PIL (Python Imaging Library)**: Used for opening, manipulating, and saving images.
- **OpenCV**: Utilized for pre-processing images, such as resizing and normalizing inputs to the model.

## FastAPI Deployment

The application is deployed using FastAPI, a high-performance web framework for building APIs with Python. FastAPI handles image uploads from users, processes them, and returns predictions from the trained model.

## Features

- **Disease Detection**: Upload an image of a potato leaf, and the system will predict if it's diseased or healthy.
- **Fast API Response**: The application is designed to provide fast and reliable predictions in real-time.

## Setup Instructions

1. Clone this repository:
   ```
   git clone https://github.com/your-username/potato-leaf-detection.git
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the FastAPI application:
   ```
   uvicorn main:app --reload
   ```

4. Open your browser and visit:
   ```
   http://127.0.0.1:8000/docs
   ```
   This will provide you with interactive API documentation to test the potato leaf detection functionality.
