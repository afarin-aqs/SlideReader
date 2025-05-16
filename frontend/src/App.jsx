import { useState } from "react";
import ImageUploader from "./ImageUploader.jsx";
import ParamEditor from "./ParamEditor.jsx";
import ImageCanvas from "./ImageCanvas.jsx";

const STAGES = {
  UPLOAD: "upload image",
  PARAMS: "tune parameters",
  EDIT: "edit circles",
};

const App = () => {
  const [stage, setStage] = useState(STAGES.UPLOAD);
  const [previewImage, setPreviewImage] = useState(null);
  const [clusterMode, setClusterMode] = useState(false);
  const [testImage, setTestImage] = useState(null);

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
    setPreviewImage(imageData);
    setStage(STAGES.PARAMS);
  };

  const handleNext = () => {
    if (stage === STAGES.PARAMS) setStage(STAGES.EDIT);
  };

  const handleBack = () => {
    if (stage === STAGES.PARAMS) setStage(STAGES.UPLOAD);
    else if (stage === STAGES.EDIT) setStage(STAGES.PARAMS);
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
          {stage === STAGES.UPLOAD && (
            <ImageUploader onImageUploaded={handleImageUploaded} />
          )}

          {stage === STAGES.PARAMS && (
            <>
              <div className="d-flex gap-2 mt-3">
                <button className="btn btn-secondary" onClick={handleBack}>
                  Back
                </button>
                <button className="btn btn-primary" onClick={handleNext}>
                  Next: Edit Circles
                </button>
              </div>
              <ParamEditor
                onImageFetched={setTestImage}
                setCircles={setCircles}
              />
            </>
          )}

          {stage === STAGES.EDIT && (
            <>
              <div className="d-flex gap-2 mt-3">
                <button className="btn btn-secondary" onClick={handleBack}>
                  Back
                </button>
              </div>
              <div className="form-check my-3">
                <input
                  className="form-check-input"
                  type="checkbox"
                  id="clusterMode"
                  checked={clusterMode}
                  onChange={(e) => setClusterMode(e.target.checked)}
                />
                <label className="form-check-label" htmlFor="clusterMode">
                  Cluster Mode
                </label>
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
          {stage === STAGES.EDIT ? (
            <ImageCanvas
              imageSrc={previewImage}
              circles={circles}
              setCircles={setCircles}
              clusterMode={clusterMode}
            />
          ) : stage === STAGES.PARAMS && testImage ? (
            <ImageCanvas
              imageSrc={testImage}
              circles={circles}
              setCircles={() => {}}
              clusterMode={false}
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
