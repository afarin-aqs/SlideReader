import React, { useState } from "react";
import ImageUploader from "./ImageUploader.jsx";
import ParamEditor from "./ParamEditor.jsx";

const App = () => {
  const [imageUploaded, setImageUploaded] = useState(false);
  const [previewImage, setPreviewImage] = useState(null);

  const handleImageUploaded = (imageData) => {
    setImageUploaded(true);
    setPreviewImage(imageData);
  };

  return (
    <div className="container-fluid">
      <div className="row">
        {/* Sidebar */}
        <div
          className="col-md-3 bg-light border-end p-3"
          style={{
            height: "100vh",
            overflowY: "auto",
          }}
        >
          <ImageUploader onImageUploaded={handleImageUploaded} />
          {imageUploaded && <ParamEditor />}
        </div>

        {/* Image display panel */}
        <div
          className="col-md-9 p-3"
          style={{
            height: "100vh",
            overflowY: "auto",
          }}
        >
          {previewImage ? (
            <img
              src={previewImage}
              alt="Preview"
              className="img-fluid"
              style={{ display: "block", margin: "0 auto" }}
            />
          ) : (
            <div
              className="text-muted d-flex justify-content-center align-items-center"
              style={{ height: "100%" }}
            >
              Upload an image first
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;
