import React from "react";
import { useParams } from "./ParamsContext.jsx";

const ParamEditor = () => {
  const { params, setParams } = useParams();

  const handleChange = (e) => {
    const { name, value } = e.target;
    setParams((prev) => ({
      ...prev,
      [name]: isNaN(value) ? value : parseFloat(value),
    }));
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
          </div>
        </div>
      </div>
    </div>
  );
};

export default ParamEditor;
