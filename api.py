import numpy as np
from flask import Flask, request, jsonify, send_file
import cv2
import base64
import io
import pickle
from flask_cors import CORS
from copy import deepcopy

from Functions import CommonFunctions, ClassesFunctions
from Classes import ScanDataObj

app = Flask(__name__)
CORS(app, expose_headers=["Content-Disposition"])

data = {
    "image": None,
    "current_filename": None,
    "params": None
}


def encode_image(image):
    """Encode a cv2 image as a Base64-encoded string"""
    success, encoded_image = cv2.imencode(".png", image)
    if not success:
        return False, None
    encoded_image = base64.b64encode(encoded_image.tobytes()).decode("utf-8")
    return True, encoded_image


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

    data["image"] = image
    filename = file.filename.strip(".tif")
    data["current_filename"] = filename

    scaled_logged_image = CommonFunctions.give_scaled_log_image(image)

    ScanDataObj.add_to_images_dict(
        file_name=filename,
        dict_key="file_image",
        dict_value=image
    )
    ScanDataObj.add_to_images_dict(
        file_name=filename,
        dict_key="file_scaled_image",
        dict_value=scaled_logged_image
    )

    success, encoded_image = encode_image(scaled_logged_image)
    if not success:
        return jsonify({"error": "Failed to encode image"}), 500

    return jsonify({"image": encoded_image})


@app.route("/set-params", methods=["POST"])
def set_params():
    """Set params"""
    params = request.get_json()
    ScanDataObj.init_or_reset_params(
        file_name=data["current_filename"],
        input_param_dict=params
    )
    data["params"] = params
    return jsonify({"status": "success"}), 200


@app.route("/get-params", methods=["GET"])
def get_params():
    """Get cached params"""
    filename = data["current_filename"]
    data_obj = ScanDataObj.get_scan_data(filename)
    if data_obj is None:
        return jsonify({"params": {}})

    params = {
        "scan_size": data_obj.scan_size,
        "assay": data_obj.assay,
        "cAb_names": data_obj.cAb_names,
        **data_obj.preprocess_params,
        **data_obj.circle_finding_params_hough,
        **data_obj.clustering_params_DBSCAN
    }
    params.pop("method_name")
    data["params"] = params

    return jsonify({"params": params})


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

    circles, clusters_ids = CommonFunctions.test_current_parameters(
        input_image=test_image, file_name=filename, plot_image=False
    )

    test_image = CommonFunctions.give_scaled_log_image(test_image)

    success, encoded_image = encode_image(test_image)
    if not success:
        return jsonify({"error": "Failed to encode image"}), 500

    return jsonify({
        "image": encoded_image,
        "circles": [circle.tolist() for circle in circles],
        "cluster_ids": clusters_ids.tolist()
    })


@app.route("/test-block-params", methods=["POST"])
def test_block_params():
    """Test block params: do block detection"""
    params = request.get_json()
    filename = data["current_filename"]

    ClassesFunctions.init_blocks_dict(
        file_name=filename,
        plot_blocks=False,
        init_offset=params["init_offset"],
        block_size_adjustment=params["block_size_adjustment"],
        block_distance_adjustment=params["block_distance_adjustment"]
    )
    borders_image = ClassesFunctions.plot_blocks_on_image(
        file_name=filename, display_in_console=False, text_color=(0, 0, 0)
    )
    scaled_logged_image = CommonFunctions.give_scaled_log_image(borders_image)

    success, encoded_image = encode_image(scaled_logged_image)
    if not success:
        return jsonify({"error": "Failed to encode image"}), 500

    return jsonify({"image": encoded_image})


@app.route("/get-pickle", methods=["GET"])
def get_pickle():
    """Send pickle file of data to frontend"""
    filename = data["current_filename"]
    data_obj = ScanDataObj.get_scan_data(filename)

    buffer = io.BytesIO()
    pickle.dump(data_obj, buffer)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"{filename}_data_obj.pickle"
    )


@app.route("/load-pickle", methods=["POST"])
def load_pickle():
    """Receive a pickle file of data from frontend and load it"""
    file = request.files["file"]
    try:
        buffer = io.BytesIO(file.read())
        data = pickle.load(buffer)
        ScanDataObj.update_scan_data_dict(data)
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
