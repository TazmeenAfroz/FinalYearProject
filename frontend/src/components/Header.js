import React from "react";
import "./Header.css";

export default function Header({ status }) {
  return (
    <header className="header">
      <div className="header__brand">
        <span className="header__logo">👁️</span>
        <h1 className="header__title">MetaGaze</h1>
        <span className="header__subtitle">Gaze Estimation</span>
      </div>
      <div className="header__status">
        <span
          className={`header__dot ${
            status === "ok" ? "header__dot--ok" : "header__dot--err"
          }`}
        />
        <span className="header__status-text">
          {status === "ok" ? "API Connected" : "API Offline"}
        </span>
      </div>
    </header>
  );
}
