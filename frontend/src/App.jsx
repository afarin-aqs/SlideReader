import { useState, useEffect } from "react";
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
  INIT: "Initialize circles and blocks data",
  EDIT: "Edit circles",
};

const STAGE_ORDER = [
  STAGES.UPLOAD,
  STAGES.PARAMS,
  STAGES.BLOCK,
  STAGES.INIT,
  STAGES.EDIT,
];

const App = () => {
  const { params, setParams, resetParams } = useParams();
  const [stage, setStage] = useState(STAGES.UPLOAD);
  const [previewImage, setPreviewImage] = useState(null);
  const [clusterMode, setClusterMode] = useState(false);
  const [testImage, setTestImage] = useState(null);
  const [initOptimizationEnabled, setInitOptimizationEnabled] = useState(false);
  const [loadingInit, setLoadingInit] = useState(false);
  const [initMessage, setInitMessage] = useState("");
  const [r, setR] = useState(0);
  const [c, setC] = useState(0);

  // Keys: id, cx, cy, r, cluster
  const [circles, setCircles] = useState([]);

  useEffect(() => {
    if (stage === STAGES.EDIT) {
      handleGetBlockData();
    }
  }, [stage, r, c]);

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

  const handleInit = async () => {
    setLoadingInit(true);
    setInitMessage("");

    try {
      await axios.post("http://127.0.0.1:5000/init-circles-clusters", {
        optimization: initOptimizationEnabled,
      });
      setInitMessage("Initial circle finding and clustering complete!");
    } catch (err) {
      console.error(err);
      alert("Failed to do initial circle finding and clustering.");
      setInitMessage("Failed to do initial circle finding and clustering.");
    } finally {
      setLoadingInit(false);
    }
  };

  const handleRowChange = (e) => {
    const value = parseInt(e.target.value);
    setR(value);
  };
  const handleColChange = (e) => {
    const value = parseInt(e.target.value);
    setC(value);
  };

  const handleGetBlockData = async () => {
    try {
      const res = await axios.get(
        `http://127.0.0.1:5000/get-block-data/r${r}c${c}`,
      );
      let { image, circles } = res.data;

      const img = `data:image/png;base64,${image}`;
      setPreviewImage(img);

      circles = circles.map(([cx, cy, r, cluster_id], index) => ({
        id: index + 1,
        cx,
        cy,
        r,
        cluster: cluster_id,
      }));
      setCircles(circles);
    } catch (error) {
      console.error("Error fetching block data:", error);
      alert("Error fetching block data");
    }
  };

  const sendEditCommand = async (cluster, command) => {
    await axios.post(`http://127.0.0.1:5000/add-edit-command/r${r}c${c}`, {
      cluster: cluster,
      command: command,
    });
    handleGetBlockData();
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

          {stage === STAGES.INIT && (
            <>
              <div className="d-flex gap-2 mt-3">
                <button
                  className="btn btn-outline-primary w-100"
                  onClick={handleInit}
                >
                  Run Initialization
                </button>
              </div>

              <div className="form-check mb-3 mt-3">
                <input
                  className="form-check-input"
                  type="checkbox"
                  id="optimizationToggle"
                  checked={initOptimizationEnabled}
                  onChange={(e) => setInitOptimizationEnabled(e.target.checked)}
                />
                <label
                  className="form-check-label"
                  htmlFor="optimizationToggle"
                >
                  Enable Optimization
                </label>
              </div>

              {loadingInit && (
                <div className="text-center mt-2">
                  <div className="spinner-border text-primary" role="status" />
                </div>
              )}
            </>
          )}

          {stage === STAGES.EDIT && (
            <>
              <div className="d-flex align-items-end gap-2 mb-3">
                <div className="col-6">
                  <label htmlFor="rowInput" className="form-label">
                    r
                  </label>
                  <input
                    type="number"
                    id="rowInput"
                    className="form-control"
                    min={0}
                    max={7}
                    value={r}
                    onChange={handleRowChange}
                  />
                </div>

                <div className="col-6">
                  <label htmlFor="colInput" className="form-label">
                    c
                  </label>
                  <input
                    type="number"
                    id="colInput"
                    className="form-control"
                    min={0}
                    max={2}
                    value={c}
                    onChange={handleColChange}
                  />
                </div>
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

              <button
                className="btn btn-primary"
                onClick={() => {
                  const clusterInput = prompt(
                    "Enter cluster ID for new circle:",
                  );
                  if (clusterInput !== null && !isNaN(clusterInput)) {
                    const clusterId = parseInt(clusterInput);
                    const newCircle = {
                      id: circles.length,
                      cx: 25,
                      cy: 25,
                      r: 15,
                      cluster: clusterId,
                    };
                    setCircles([...circles, newCircle]);
                    sendEditCommand(clusterId, "add 1 to abs");
                  }
                }}
              >
                Add Circle
              </button>

              <small className="text-muted d-block mt-1">
                Double click a circle to delete
              </small>
            </>
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
              sendEditCommand={sendEditCommand}
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
          ) : stage === STAGES.INIT ? (
            <>
              {initMessage && (
                <div className="alert alert-success mt-3" role="alert">
                  {initMessage}
                </div>
              )}
            </>
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
