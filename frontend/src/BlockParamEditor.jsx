import { useState, useEffect } from "react";
import axios from "axios";

const BlockParamEditor = ({ onImageFetched }) => {
  const [params, setParams] = useState({
    init_offset: "0,0",
    block_size_adjustment: 0,
    block_distance_adjustment: "0,0",
  });
  const [message, setMessage] = useState(null);

  useEffect(() => {
    syncParamsAndFetchImage();
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setParams((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const parseParamValue = (key, value) => {
    if (key === "init_offset" || key === "block_distance_adjustment") {
      return value
        .split(",")
        .map((v) => parseFloat(v.trim()))
        .filter((v) => !isNaN(v));
    } else {
      return parseFloat(value);
    }
  };

  const syncParamsAndFetchImage = async () => {
    const parsedParams = {};
    for (const [key, value] of Object.entries(params)) {
      parsedParams[key] = parseParamValue(key, value);
    }

    try {
      const res = await axios.post(
        "http://127.0.0.1:5000/test-block-params",
        parsedParams,
      );
      const { image } = res.data;
      const img = `data:image/png;base64,${image}`;
      onImageFetched(img);

      setMessage("Block detection parameters saved successfully!");
      setTimeout(() => setMessage(null), 3000);
    } catch (error) {
      console.error("Error saving or testing block params:", error);
      alert("Failed to save or test block parameters.");
    }
  };

  return (
    <div className="p-3">
      {Object.entries(params).map(([key, value]) => (
        <div className="mb-3" key={key}>
          <label htmlFor={key} className="form-label">
            {key}
          </label>
          <input
            type="text"
            className="form-control"
            id={key}
            name={key}
            value={value}
            onChange={handleChange}
          />
        </div>
      ))}

      <div className="d-flex justify-content-center mt-3">
        <button className="btn btn-primary" onClick={syncParamsAndFetchImage}>
          Save Parameters
        </button>
      </div>

      {message && (
        <div
          className="alert alert-success alert-dismissible fade show position-fixed bottom-0 start-50 translate-middle-x mb-3"
          role="alert"
          style={{ zIndex: 100, minWidth: "300px" }}
        >
          {message}
          <button
            type="button"
            className="btn-close"
            aria-label="Close"
            onClick={() => setMessage(null)}
          ></button>
        </div>
      )}
    </div>
  );
};

export default BlockParamEditor;
