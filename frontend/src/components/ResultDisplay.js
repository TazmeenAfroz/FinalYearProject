import React from "react";
import "./ResultDisplay.css";

export default function ResultDisplay({ imageUrl }) {
  if (!imageUrl) return null;

  const handleDownload = async () => {
    try {
      const res = await fetch(imageUrl);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "metagaze_result.png";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch {
      alert("Failed to download the image.");
    }
  };

  return (
    <div className="result">
      <h3 className="result__title">Annotated Output</h3>
      <div className="result__frame">
        <img src={imageUrl} alt="Gaze annotated" className="result__image" />
      </div>
      <button onClick={handleDownload} className="result__download">
        ⬇ Download Image
      </button>
    </div>
  );
}
