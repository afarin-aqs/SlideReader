from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/convert-tif-to-png", methods=["POST"])
def convert_tif_to_png():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if not file.filename.endswith(".tif"):
        return jsonify({"error": "Only .tif files are supported"}), 400

    image = Image.open(file)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return jsonify({"image": encoded_image})

if __name__ == "__main__":
    app.run(debug=True)
