import { useState } from "react";
import axios from "axios";
import { saveAs } from "file-saver";
import Uploader from "./Uploader.jsx";
import ParamEditor from "./ParamEditor.jsx";
import BlockParamEditor from "./BlockParamEditor.jsx";
import ImageCanvas from "./ImageCanvas.jsx";
import { useParams } from "./ParamsContext.jsx";

const STAGES = {
  UPLOAD: "Upload an image",
  PARAMS: "Tune circle/cluster params",
  BLOCK: "Tune block detection params",
  EDIT: "Edit circles",
};

const STAGE_ORDER = [STAGES.UPLOAD, STAGES.PARAMS, STAGES.BLOCK, STAGES.EDIT];

const App = () => {
  const { params, setParams, resetParams } = useParams();
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

  const handleSavePickle = async () => {
    try {
      const response = await axios.get("http://127.0.0.1:5000/get-pickle", {
        responseType: "blob",
      });
      const disposition = response.headers["content-disposition"];
      const match = disposition?.match(/filename="?([^"]+)"?/);
      const filename = match?.[1] || "scan_data.pickle";
      saveAs(response.data, filename);
    } catch (error) {
      console.error("Error downloading pickle file:", error);
      alert("Failed to download pickle file.");
    }
  };

  const handleReset = async () => {
    await axios.get("http://127.0.0.1:5000/reset");
    resetParams();
    setStage(STAGES.UPLOAD);
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
          <div className="mb-3">
            <h6>
              {currentStageIndex + 1}. {stage}
            </h6>
          </div>

          {currentStageIndex > 0 && (
            <div className="d-flex gap-2">
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
            <Uploader onImageUploaded={handleImageUploaded} />
          )}

          {stage === STAGES.PARAMS && (
            <ParamEditor
              onImageFetched={setTestImage}
              setCircles={setCircles}
            />
          )}

          {stage === STAGES.BLOCK && (
            <BlockParamEditor onImageFetched={setTestImage} />
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

          {stage !== STAGES.UPLOAD && (
            <div className="mt-3">
              <div className="d-flex gap-2">
                <button
                  className="btn btn-outline-primary w-100"
                  onClick={handleReset}
                >
                  Reset
                </button>
                <button
                  className="btn btn-outline-success w-100"
                  onClick={handleSavePickle}
                >
                  Save Pickle
                </button>
              </div>
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
          ) : stage === STAGES.BLOCK && testImage ? (
            <div
              style={{
                height: "100%",
                overflowY: "auto",
              }}
            >
              <img
                src={testImage}
                style={{
                  width: "100%",
                }}
              />
            </div>
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
