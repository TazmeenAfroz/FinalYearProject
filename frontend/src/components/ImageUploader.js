import React, { useRef, useState, useCallback, useEffect } from "react";
import "./ImageUploader.css";

export default function ImageUploader({ onFileSelected, loading, resetKey }) {
  const inputRef = useRef();
  const [preview, setPreview] = useState(null);
  const [dragOver, setDragOver] = useState(false);

  useEffect(() => {
    setPreview(null);
    if (inputRef.current) inputRef.current.value = "";
  }, [resetKey]);

  const handleFile = useCallback(
    (file) => {
      if (!file) return;
      if (!file.type.startsWith("image/")) {
        alert("Please select a valid image file.");
        return;
      }
      setPreview(URL.createObjectURL(file));
      onFileSelected(file);
    },
    [onFileSelected]
  );

  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    handleFile(file);
  };

  const onDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const onDragLeave = () => setDragOver(false);

  return (
    <div className="uploader">
      <div
        className={`uploader__dropzone ${dragOver ? "uploader__dropzone--active" : ""}`}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onClick={() => inputRef.current?.click()}
      >
        {preview ? (
          <img src={preview} alt="Preview" className="uploader__preview" />
        ) : (
          <div className="uploader__placeholder">
            <span className="uploader__icon">📷</span>
            <p>Drag & drop an image here</p>
            <p className="uploader__hint">or click to browse</p>
          </div>
        )}
      </div>

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="uploader__input"
        onChange={(e) => handleFile(e.target.files?.[0])}
      />

      {loading && (
        <div className="uploader__loading">
          <div className="uploader__spinner" />
          <span>Running inference…</span>
        </div>
      )}
    </div>
  );
}
