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

  return (
    <div className="accordion mt-3" id="paramAccordion">
      <div className="accordion-item">
        <h2 className="accordion-header" id="headingParams">
          <button
            className="accordion-button"
            type="button"
            data-bs-toggle="collapse"
            data-bs-target="#collapseParams"
            aria-expanded="true"
            aria-controls="collapseParams"
          >
            Edit Parameters
          </button>
        </h2>
        <div
          id="collapseParams"
          className="accordion-collapse collapse show"
          aria-labelledby="headingParams"
          data-bs-parent="#paramAccordion"
        >
          <div className="accordion-body">
            {Object.entries(params).map(([key, value]) => (
              <div className="mb-3" key={key}>
                <label htmlFor={key} className="form-label">
                  {key}
                </label>
                <input
                  type={typeof value === "number" ? "number" : "text"}
                  className="form-control"
                  step="any"
                  name={key}
                  id={key}
                  value={value}
                  onChange={handleChange}
                />
              </div>
            ))}

            <button className="btn btn-primary mt-3" onClick={handleSave}>
              Save Parameters
            </button>

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
        </div>
      </div>
    </div>
  );
};

export default ParamEditor;
