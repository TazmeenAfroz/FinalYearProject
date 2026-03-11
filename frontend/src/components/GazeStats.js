import React from "react";
import "./GazeStats.css";

export default function GazeStats({ data }) {
  if (!data) return null;

  const rows = [
    { label: "Gaze Pitch", value: `${data.pitch_deg?.toFixed(2)}°` },
    { label: "Gaze Yaw",   value: `${data.yaw_deg?.toFixed(2)}°` },
    { label: "Head Pitch",  value: `${data.head_pitch_deg?.toFixed(2)}°` },
    { label: "Head Yaw",    value: `${data.head_yaw_deg?.toFixed(2)}°` },
  ];

  const vec = data.gaze_vector;

  return (
    <div className="stats">
      <h3 className="stats__title">Gaze Estimation Results</h3>

      <div className="stats__grid">
        {rows.map((r) => (
          <div key={r.label} className="stats__card">
            <span className="stats__label">{r.label}</span>
            <span className="stats__value">{r.value}</span>
          </div>
        ))}
      </div>

      {vec && (
        <div className="stats__vector">
          <span className="stats__label">3D Gaze Vector</span>
          <code className="stats__code">
            [{vec[0]?.toFixed(4)}, {vec[1]?.toFixed(4)}, {vec[2]?.toFixed(4)}]
          </code>
        </div>
      )}
    </div>
  );
}
