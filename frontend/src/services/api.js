import axios from "axios";

const API_BASE = process.env.REACT_APP_API_URL || "";

const api = axios.create({
  baseURL: API_BASE,
  timeout: 60000,
});

export const getHealth = () => api.get("/api/health");

export const getModelInfo = () => api.get("/api/model-info");

export const getModels = async () => {
  const res = await api.get("/api/models");
  return res.data;
};

export const predictGaze = async (file, modelName) => {
  const form = new FormData();
  form.append("file", file);
  const params = modelName ? { model: modelName } : {};
  const res = await api.post("/api/predict", form, {
    headers: { "Content-Type": "multipart/form-data" },
    params,
  });
  return res.data;
};

export const predictAnnotated = async (file) => {
  const form = new FormData();
  form.append("file", file);
  const res = await api.post("/api/predict/annotated", form, {
    responseType: "blob",
  });
  return URL.createObjectURL(res.data);
};

export const getResultImageUrl = (resultId) =>
  `${API_BASE}/api/result/${resultId}`;

export default api;
