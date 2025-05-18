import { useState } from "react";
import ImageUploader from "./ImageUploader.jsx";
import ParamEditor from "./ParamEditor.jsx";
import ImageCanvas from "./ImageCanvas.jsx";

const STAGES = {
  UPLOAD: "upload image",
  PARAMS: "tune parameters",
  BLOCK: "detect blocks",
  EDIT: "edit circles",
};

const STAGE_ORDER = [STAGES.UPLOAD, STAGES.PARAMS, STAGES.BLOCK, STAGES.EDIT];

const App = () => {
  const [stage, setStage] = useState(STAGES.UPLOAD);
  const [previewImage, setPreviewImage] = useState(null);
  const [clusterMode, setClusterMode] = useState(false);
  const [testImage, setTestImage] = useState(null);

  // Keys: id, cx, cy, r, cluster
  const [circles, setCircles] = useState([]);

  const currentStageIndex = STAGE_ORDER.indexOf(stage);

  const handleImageUploaded = (imageData) => {
    setPreviewImage(imageData);
    setStage(STAGES.PARAMS);
  };

  const handleNext = () => {
    if (currentStageIndex < STAGE_ORDER.length - 1) {
      setStage(STAGE_ORDER[currentStageIndex + 1]);
    }
  };

  const handleBack = () => {
    if (currentStageIndex > 0) {
      setStage(STAGE_ORDER[currentStageIndex - 1]);
    }
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
          {currentStageIndex > 0 && (
            <div className="d-flex gap-2 mt-4">
              <button className="btn btn-secondary" onClick={handleBack}>
                Back
              </button>
              {stage !== STAGES.UPLOAD &&
                currentStageIndex < STAGE_ORDER.length - 1 && (
                  <button className="btn btn-primary" onClick={handleNext}>
                    Next
                  </button>
                )}
            </div>
          )}

          {stage === STAGES.UPLOAD && (
            <ImageUploader onImageUploaded={handleImageUploaded} />
          )}

          {stage === STAGES.PARAMS && (
            <ParamEditor
              onImageFetched={setTestImage}
              setCircles={setCircles}
            />
          )}

          {stage === STAGES.BLOCK && (
            <div className="mt-4">
              <h5>Block Detection Parameters</h5>
            </div>
          )}

          {stage === STAGES.EDIT && (
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
          ) : stage === STAGES.BLOCK ? (
            <p>Block Detection</p>
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
