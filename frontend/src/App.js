import React, { useState, useEffect, useCallback } from "react";
import Header from "./components/Header";
import ImageUploader from "./components/ImageUploader";
import LiveCamera from "./components/LiveCamera";
import GazeStats from "./components/GazeStats";
import ResultDisplay from "./components/ResultDisplay";
import { getHealth, getModels, predictGaze, getResultImageUrl } from "./services/api";
import "./App.css";

export default function App() {
  const [apiStatus, setApiStatus] = useState("unknown");
  const [loading, setLoading] = useState(false);
  const [gazeData, setGazeData] = useState(null);
  const [annotatedUrl, setAnnotatedUrl] = useState(null);
  const [error, setError] = useState(null);

  const [resetKey, setResetKey] = useState(0);

  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [mode, setMode] = useState("image");

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        await getHealth();
        if (!cancelled) setApiStatus("ok");
      } catch {
        if (!cancelled) setApiStatus("error");
      }
    };
    check();
    const interval = setInterval(check, 15000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const data = await getModels();
        setModels(data.models || []);
        setSelectedModel(data.default || "");
      } catch {
      }
    };
    fetchModels();
  }, []);

  const handleReset = useCallback(() => {
    setGazeData(null);
    setAnnotatedUrl(null);
    setError(null);
    setResetKey((k) => k + 1);
  }, []);

  const handleFile = useCallback(async (file) => {
    setLoading(true);
    setError(null);
    setGazeData(null);
    setAnnotatedUrl(null);

    try {
      const result = await predictGaze(file, selectedModel || undefined);
      setGazeData(result);
      if (result.result_id) {
        setAnnotatedUrl(getResultImageUrl(result.result_id));
      }
    } catch (err) {
      const msg =
        err.response?.data?.detail ||
        err.message ||
        "Something went wrong.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [selectedModel]);

  return (
    <div className="app">
      <Header status={apiStatus} />

      <main className="app__main">
        <div className="app__mode-toggle">
          <button
            className={`app__mode-btn ${mode === "image" ? "app__mode-btn--active" : ""}`}
            onClick={() => setMode("image")}
          >
            📷 Image Upload
          </button>
          <button
            className={`app__mode-btn ${mode === "live" ? "app__mode-btn--active" : ""}`}
            onClick={() => setMode("live")}
          >
            🎥 Live Camera
          </button>
        </div>

        {models.length > 0 && (
          <div className="app__model-selector">
            <label htmlFor="model-select">Model: </label>
            <select
              id="model-select"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={loading}
            >
              {models.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>
        )}

        {mode === "image" && (
          <>
            <section className="app__section">
              <h2 className="app__heading">Upload Image</h2>
              <p className="app__desc">
                Upload a face image to estimate gaze direction using the GazeSymCAT
                transformer model trained on ETH-XGaze.
              </p>
              <ImageUploader onFileSelected={handleFile} loading={loading} resetKey={resetKey} />
              {gazeData && (
                <button className="app__reset-btn" onClick={handleReset}>
                  🔄 Try Another Image
                </button>
              )}
            </section>

            {error && (
              <div className="app__error">
                <strong>Error:</strong> {error}
              </div>
            )}

            {gazeData && (
              <>
                <section className="app__section">
                  <GazeStats data={gazeData} />
                </section>

                <section className="app__section">
                  <ResultDisplay imageUrl={annotatedUrl} />
                </section>
              </>
            )}
          </>
        )}

        {mode === "live" && (
          <section className="app__section">
            <h2 className="app__heading">Live Camera</h2>
            <p className="app__desc">
              Real-time gaze estimation from your webcam feed. Click start to begin.
            </p>
            <LiveCamera selectedModel={selectedModel} />
          </section>
        )}
      </main>

      <footer className="app__footer">
        MetaGaze &middot; GazeSymCAT &middot; ETH-XGaze
      </footer>
    </div>
  );
}
