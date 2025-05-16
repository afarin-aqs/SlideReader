import { useState } from "react";
import axios from "axios";

const ImageUploader = ({ onImageUploaded }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setError("");
    setLoading(true);

    const formData = new FormData();
    formData.append("image", file);

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/prep-image",
        formData,
      );
      const imageBase64 = response.data.image;
      const imageUrl = `data:image/png;base64,${imageBase64}`;
      onImageUploaded(imageUrl);
    } catch (err) {
      console.error(err);
      setError("Failed to upload or process image.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mt-3">
      <h5>Upload .tif Image</h5>
      <input
        type="file"
        accept=".tif"
        onChange={handleUpload}
        className="form-control mb-3"
      />
      {loading && (
        <div className="text-center">
          <div className="spinner-border text-primary" role="status" />
        </div>
      )}
      {error && <div className="alert alert-danger mt-2">{error}</div>}
    </div>
  );
};

export default ImageUploader;
