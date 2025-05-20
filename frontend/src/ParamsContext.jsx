import { createContext, useState, useContext } from "react";

const ParamsContext = createContext();

const defaultParams = {
  scan_size: 5,
  assay: "OF",
  cAb_names: [],
  blur_kernel_size: 11,
  contrast_thr: 200,
  canny_edge_thr1: 90,
  canny_edge_thr2: 290,
  dp: 1.6,
  param1: 11,
  param2: 22,
  minRadius: 11,
  maxRadius: 18,
  eps: 500,
  x_power: 3,
  y_power: 5,
  min_samples: 3,
};

export const ParamsProvider = ({ children }) => {
  const [params, setParams] = useState(defaultParams);

  const resetParams = () => {
    setParams(defaultParams);
  };

  return (
    <ParamsContext.Provider value={{ params, setParams, resetParams }}>
      {children}
    </ParamsContext.Provider>
  );
};

export const useParams = () => useContext(ParamsContext);
