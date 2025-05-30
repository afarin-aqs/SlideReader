import base64
import io
import pickle
from copy import deepcopy

import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from Functions import CommonFunctions, ClassesFunctions
from Classes import ScanDataObj

app = Flask(__name__)
CORS(app, expose_headers=["Content-Disposition"])

default_block_params = {
    "init_offset": [0, 0],
    "block_size_adjustment": 0,
    "block_distance_adjustment": [0, 0]
}

data = {
    "image": None,
    "current_filename": None,
    "params": None,
    "block_params": default_block_params,
    "edit_commands": {}
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
    data["params"] = None
    data["block_params"] = default_block_params

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
    data_obj = ScanDataObj.get_scan_data(filename)

    old_params = data["block_params"]
    # Initialize blocks only when no data or params have changed
    if data_obj.blocks_dict == {} or params != old_params:
        print("Saving block params and initializing blocks")
        data["block_params"] = params
        ClassesFunctions.init_blocks_dict(
            file_name=filename,
            plot_blocks=False,
            init_offset=params["init_offset"],
            block_size_adjustment=params["block_size_adjustment"],
            block_distance_adjustment=params["block_distance_adjustment"]
        )
    for block in data_obj.blocks_dict.values():
        block.add_cropped_images()
    borders_image = ClassesFunctions.plot_blocks_on_image(
        file_name=filename, display_in_console=False, text_color=(0, 0, 0)
    )
    scaled_logged_image = CommonFunctions.give_scaled_log_image(borders_image)

    success, encoded_image = encode_image(scaled_logged_image)
    if not success:
        return jsonify({"error": "Failed to encode image"}), 500

    return jsonify({"image": encoded_image})


@app.route("/init-circles-clusters", methods=["POST"])
def init_circles_clusters():
    """Initialize circles and clusters"""
    optimize_spots_coords = request.get_json().get("optimization")

    filename = data["current_filename"]
    data_obj = ScanDataObj.get_scan_data(filename)

    CommonFunctions.do_initial_circle_finding(filename)
    ClassesFunctions.init_clusters_dict(
        data_obj.sorted_circles,
        data_obj.predicted_clusters_ids,
        filename,
        optimize_spots_coords=optimize_spots_coords
    )
    ClassesFunctions.connect_clusters_to_blocks(filename, debug_blocks=[])

    return jsonify({"status": "success"}), 200


@app.route("/get-block-data/r<int:r>c<int:c>", methods=["GET"])
def get_block_data(r, c):
    """Get circles and clusters data of block at specified row and column"""
    filename = data["current_filename"]

    data_obj = ScanDataObj.get_scan_data(filename)
    block = data_obj.get_block(f"r{r}c{c}")
    scaled_image = ScanDataObj.get_block_image(
        filename, block.block_id, image_tag="scaled_image"
    )
    colored_image = CommonFunctions.make_3D_image(scaled_image)
    success, encoded_image = encode_image(colored_image)

    all_circles = []
    for cluster_id in block.clusters_ids_list:
        cluster = data_obj.get_cluster(cluster_id)
        cluster.sort_coords_lists()
        circles = cluster.spots_coords_in_block_list
        circles = [
            np.append(arr, cluster_id) for arr in circles
        ]  # Add cluster_id to x, y, r
        all_circles.extend(circles)

    return jsonify({
        "image": encoded_image,
        "circles": [circle.tolist() for circle in all_circles],
    })


@app.route("/add-edit-command/r<int:r>c<int:c>", methods=["POST"])
def add_edit_command(r, c):
    """Add edit command for specified block and edit block with all commands"""
    command = request.get_json()["command"]
    cluster = request.get_json()["cluster"]
    if f"r{r}c{c}" not in data["edit_commands"]:
        data["edit_commands"][f"r{r}c{c}"] = {}
    if cluster not in data["edit_commands"][f"r{r}c{c}"]:
        data["edit_commands"][f"r{r}c{c}"][cluster] = []
    data["edit_commands"][f"r{r}c{c}"][cluster].append(command)

    filename = data["current_filename"]
    data_obj = ScanDataObj.get_scan_data(filename)
    block = data_obj.get_block(f"r{r}c{c}")
    block.edit_block(
        manual_spot_edit_dict=data["edit_commands"].get(f"r{r}c{c}"),
        plot_before_after=False,
        overwrite=True,
        with_restore=True
    )

    return jsonify({"status": "success"}), 200


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


@app.route("/reset", methods=["GET"])
def reset():
    """Reset"""
    ScanDataObj.all_scan_data = {}
    ScanDataObj.images_dict = {}

    data = {
        "image": None,
        "current_filename": None,
        "params": None,
        "block_params": default_block_params,
        "edit_commands": {}
    }

    return jsonify({"status": "success"}), 200


if __name__ == "__main__":
    app.run(debug=True)
