import requests
import os

MODEL_API_URL = "http://127.0.0.1:8000"

def predict_digit_api(canvas_img_data):
    data_bytes = canvas_img_data.tobytes()
    files = {'file': ('data.npy', data_bytes, 'application/octet-stream')}
    response = requests.post(f"{MODEL_API_URL}/predict", files=files)
    print("RESPONSE ", response)
    return response.json()
