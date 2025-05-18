import { useState } from "react";
import axios from "axios";

const Uploader = ({ onImageUploaded }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleUpload = async (e) => {
    const files = Array.from(e.target.files);
    const imageFile = files.find((f) => f.name.endsWith(".tif"));
    const pickleFile = files.find((f) => f.name.endsWith(".pickle"));

    if (!imageFile) {
      setError("Please select a .tif image file.");
      return;
    }

    setError("");
    setLoading(true);

    try {
      const imageFormData = new FormData();
      imageFormData.append("image", imageFile);

      const imageRes = await axios.post(
        "http://127.0.0.1:5000/prep-image",
        imageFormData,
      );

      const imageBase64 = imageRes.data.image;
      const imageUrl = `data:image/png;base64,${imageBase64}`;
      onImageUploaded(imageUrl);

      if (pickleFile) {
        const pickleFormData = new FormData();
        pickleFormData.append("file", pickleFile);

        await axios.post("http://127.0.0.1:5000/load-pickle", pickleFormData);
      }
    } catch (err) {
      console.error(err);
      setError("Failed to upload or process files.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mt-3">
      <h5>Upload .tif Image and Optional .pickle File</h5>
      <input
        type="file"
        multiple
        accept=".tif,.pickle"
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

export default Uploader;
