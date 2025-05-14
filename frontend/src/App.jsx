import React, { useState } from "react";
import ImageUploader from "./ImageUploader.jsx";
import ParamEditor from "./ParamEditor.jsx";
import ImageCanvas from "./ImageCanvas.jsx";

const App = () => {
  const [imageUploaded, setImageUploaded] = useState(false);
  const [previewImage, setPreviewImage] = useState(null);

  const [circles, setCircles] = useState([
    { id: 1, cx: 400, cy: 1000, r: 80, cluster: 1 },
    { id: 2, cx: 750, cy: 2000, r: 85, cluster: 1 },
    { id: 3, cx: 1100, cy: 3000, r: 90, cluster: 1 },
    { id: 4, cx: 1450, cy: 4000, r: 80, cluster: 2 },
    { id: 5, cx: 1800, cy: 5000, r: 85, cluster: 2 },
    { id: 6, cx: 2150, cy: 6000, r: 90, cluster: 2 },
    { id: 7, cx: 2500, cy: 7000, r: 80, cluster: 3 },
    { id: 8, cx: 2850, cy: 8000, r: 85, cluster: 3 },
    { id: 9, cx: 3200, cy: 9000, r: 90, cluster: 3 },
    { id: 10, cx: 3550, cy: 10000, r: 80, cluster: -1 },
  ]);

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
          {imageUploaded && (
            <>
              {/* ParamEditor */}
              <ParamEditor />

              <hr className="my-3" />

              {/* Circles Info Accordion */}
              <div className="accordion" id="circlesAccordion">
                <div className="accordion-item">
                  <h2 className="accordion-header" id="headingCircles">
                    <button
                      className="accordion-button"
                      type="button"
                      data-bs-toggle="collapse"
                      data-bs-target="#collapseCircles"
                      aria-expanded="true"
                      aria-controls="collapseCircles"
                    >
                      Detected Circles
                    </button>
                  </h2>
                  <div
                    id="collapseCircles"
                    className="accordion-collapse collapse show"
                    aria-labelledby="headingCircles"
                    data-bs-parent="#circlesAccordion"
                  >
                    <div className="accordion-body">
                      {circles.map((c) => (
                        <div key={c.id} className="mb-3 p-2 border rounded">
                          <div>
                            <strong>Circle {c.id}</strong>
                          </div>
                          <div>X: {Math.round(c.cx)}</div>
                          <div>Y: {Math.round(c.cy)}</div>
                          <div>Radius: {Math.round(c.r)}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </>
          )}
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
            <ImageCanvas
              imageSrc={previewImage}
              circles={circles}
              setCircles={setCircles}
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
