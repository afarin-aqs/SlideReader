import numpy as np
from flask import Flask, request, jsonify
import cv2
import base64
from flask_cors import CORS
from copy import deepcopy

from Functions.CommonFunctions import give_scaled_log_image, test_current_parameters
from Classes.ScanDataObj import create_new_scan_data, init_or_reset_params

app = Flask(__name__)
CORS(app)

data = {
    "image": None,
    "current_filename": None,
    "params": None
}


@app.route("/prep-image", methods=["POST"])
def scale_log_convert_image():
    """Log and normalize image, then convert tif to png
       Also initializes a ScanDataObj and saves image for future use
    """
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if not file.filename.endswith(".tif"):
        return jsonify({"error": "Only .tif files are supported"}), 400

    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

    data["image"] = image
    data["current_filename"] = file.filename

    image = give_scaled_log_image(image)

    success, encoded_image = cv2.imencode(".png", image)
    if not success:
        return jsonify({"error": "Failed to encode image"}), 500

    encoded_image = base64.b64encode(encoded_image.tobytes()).decode("utf-8")

    return jsonify({"image": encoded_image})


@app.route("/set-params", methods=["POST"])
def set_params():
    """Set params"""
    params = request.get_json()
    init_or_reset_params(
        file_name=data["current_filename"],
        input_param_dict=params
    )
    data["params"] = params
    return jsonify({"status": "success"}), 200


@app.route("/test-params", methods=["GET"])
def test_params():
    """Test params: do circle and cluster detection on a cropped image"""
    image = data["image"]
    params = data["params"]
    filename = data["current_filename"]

    if params["assay"] == "SD4":
        test_image = deepcopy(image)[1000:1700, 500:2500]
    else:
        test_image = deepcopy(image)[500:1400, 50:2600]

    circles, clusters_ids = test_current_parameters(
        input_image=test_image, file_name=filename, plot_image=False
    )

    test_image = give_scaled_log_image(test_image)

    success, encoded_image = cv2.imencode(".png", test_image)
    if not success:
        return jsonify({"error": "Failed to encode image"}), 500
    encoded_image = base64.b64encode(encoded_image.tobytes()).decode("utf-8")

    return jsonify({
        "image": encoded_image,
        "circles": [circle.tolist() for circle in circles],
        "cluster_ids": clusters_ids.tolist()
    })


if __name__ == "__main__":
    app.run(debug=True)
