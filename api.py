import numpy as np
from flask import Flask, request, jsonify
import cv2
import base64
from flask_cors import CORS

from Functions.CommonFunctions import give_scaled_log_image

app = Flask(__name__)
CORS(app)

@app.route("/prep-image", methods=["POST"])
def scale_log_convert_image():
    """Log and normalize image, then convert tif to png"""
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if not file.filename.endswith(".tif"):
        return jsonify({"error": "Only .tif files are supported"}), 400

    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

    image = give_scaled_log_image(image)

    success, encoded_image = cv2.imencode(".png", image)
    if not success:
        return jsonify({"error": "Failed to encode image"}), 500

    encoded_image_base64 = base64.b64encode(encoded_image.tobytes()).decode("utf-8")

    return jsonify({"image": encoded_image_base64})

if __name__ == "__main__":
    app.run(debug=True)
