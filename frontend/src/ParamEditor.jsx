import { useState } from "react";
import axios from "axios";
import { useParams } from "./ParamsContext.jsx";

const ParamEditor = ({ onImageFetched, setCircles }) => {
  const { params, setParams } = useParams();
  const [message, setMessage] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setParams((prev) => ({
      ...prev,
      [name]: isNaN(value) ? value : parseFloat(value),
    }));
  };

  const handleSave = async () => {
    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/set-params",
        params,
      );
      setMessage("Parameters saved successfully!");
      setTimeout(() => {
        setMessage(null);
      }, 3000);
    } catch (error) {
      console.error("Error saving parameters:", error);
      alert("An error occurred while saving parameters.");
    }

    try {
      const res = await axios.get("http://127.0.0.1:5000/test-params");
      let { image, circles, cluster_ids } = res.data;

      const img = `data:image/png;base64,${image}`;
      onImageFetched(img);

      circles = circles.map(([cx, cy, r], index) => ({
        id: index + 1,
        cx,
        cy,
        r,
        cluster: cluster_ids[index] ?? -1,
      }));
      setCircles(circles);
    } catch (error) {
      console.log("Error fetching test image: ", error);
      alert("An error occurred while fetching test image.");
    }
  };

  const paramEntries = Object.entries(params);
  const paramPairs = [];
  for (let i = 0; i < paramEntries.length; i += 2) {
    paramPairs.push(paramEntries.slice(i, i + 2));
  }

  return (
    <div className="p-3">
      {paramPairs.map((pair, rowIndex) => (
        <div className="row mb-3" key={rowIndex}>
          {pair.map(([key, value], colIndex) => (
            <div className="col-6" key={key}>
              <label
                htmlFor={key}
                className={`form-label d-block ${
                  colIndex === 0 ? "text-start" : "text-end"
                }`}
              >
                {key}
              </label>
              <input
                type={typeof value === "number" ? "number" : "text"}
                className="form-control form-control-sm"
                step="any"
                name={key}
                id={key}
                value={value}
                onChange={handleChange}
              />
            </div>
          ))}
        </div>
      ))}

      <div className="d-flex justify-content-center mt-3">
        <button className="btn btn-primary" onClick={handleSave}>
          Save Parameters
        </button>
      </div>

      {message && (
        <div
          className="alert alert-success alert-dismissible fade show position-fixed top-0 start-50 translate-middle-x mt-3"
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

export default ParamEditor;
