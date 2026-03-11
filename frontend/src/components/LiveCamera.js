import React, { useRef, useState, useEffect, useCallback } from "react";
import "./LiveCamera.css";

const WS_URL = process.env.REACT_APP_API_URL
  ? process.env.REACT_APP_API_URL.replace(/^http/, "ws") + "/api/ws/live"
  : `ws://${window.location.hostname}:8000/api/ws/live`;

export default function LiveCamera({ selectedModel }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const frameLoopRef = useRef(null);

  const [running, setRunning] = useState(false);
  const [annotatedSrc, setAnnotatedSrc] = useState(null);
  const [gazeData, setGazeData] = useState(null);
  const [error, setError] = useState(null);
  const [fps, setFps] = useState(0);
  const lastFrameTime = useRef(Date.now());
  const busyRef = useRef(false);
  const frameSkipCounterRef = useRef(0);
  const [useLowQuality, setUseLowQuality] = useState(false);

  const stopStream = useCallback(() => {
    if (frameLoopRef.current) {
      cancelAnimationFrame(frameLoopRef.current);
      frameLoopRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    busyRef.current = false;
    setRunning(false);
    setAnnotatedSrc(null);
    setGazeData(null);
    setFps(0);
  }, []);

  useEffect(() => {
    return () => stopStream();
  }, [stopStream]);

  const startStream = useCallback(async () => {
    setError(null);

    try {
      const healthRes = await fetch('/api/health');
      const healthData = await healthRes.json();
      if (healthData.is_cpu) {
        setUseLowQuality(true);
      }
    } catch (e) {
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: 640, height: 480 },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setRunning(true);
        busyRef.current = false;
        const loop = () => {
          frameLoopRef.current = requestAnimationFrame(loop);

          if (busyRef.current) {
            frameSkipCounterRef.current++;
            if (frameSkipCounterRef.current < 2) return;
            frameSkipCounterRef.current = 0;
          }

          const video = videoRef.current;
          const canvas = canvasRef.current;
          const socket = wsRef.current;
          if (!video || !canvas || !socket || socket.readyState !== WebSocket.OPEN) return;
          if (!video.videoWidth || !video.videoHeight) return;

          if (useLowQuality) {
            canvas.width = 320;
            canvas.height = 240;
            const ctx = canvas.getContext("2d");
            ctx.drawImage(video, 0, 0, 320, 240);
            const frameB64 = canvas.toDataURL("image/jpeg", 0.5);
            busyRef.current = true;
            frameSkipCounterRef.current = 0;
            socket.send(JSON.stringify({ frame: frameB64, model: selectedModel }));
          } else {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext("2d");
            ctx.drawImage(video, 0, 0);
            const frameB64 = canvas.toDataURL("image/jpeg", 0.92);
            busyRef.current = true;
            frameSkipCounterRef.current = 0;
            socket.send(JSON.stringify({ frame: frameB64, model: selectedModel }));
          }
        };
        frameLoopRef.current = requestAnimationFrame(loop);
      };

      ws.onmessage = (evt) => {
        const data = JSON.parse(evt.data);
        busyRef.current = false;

        if (data.error) {
          return;
        }
        if (data.annotated_frame) {
          setAnnotatedSrc(data.annotated_frame);
        }
        setGazeData({
          pitch_deg: data.pitch_deg,
          yaw_deg: data.yaw_deg,
          head_pitch_deg: data.head_pitch_deg,
          head_yaw_deg: data.head_yaw_deg,
          gaze_vector: data.gaze_vector,
        });
        const now = Date.now();
        setFps(Math.round(1000 / (now - lastFrameTime.current)));
        lastFrameTime.current = now;
      };

      ws.onerror = () => {
        setError("WebSocket connection error");
        stopStream();
      };

      ws.onclose = () => {
        setRunning(false);
      };
    } catch (e) {
      setError("Could not access camera: " + e.message);
    }
  }, [selectedModel, stopStream]);

  return (
    <div className="live">
      <canvas ref={canvasRef} className="live__canvas-hidden" />

      <div className="live__controls">
        {!running ? (
          <button className="live__btn live__btn--start" onClick={startStream}>
            🎥 Start Live Camera
          </button>
        ) : (
          <button className="live__btn live__btn--stop" onClick={stopStream}>
            ⏹ Stop Camera
          </button>
        )}
        {running && <span className="live__fps">{fps} FPS</span>}
      </div>

      {error && (
        <div className="live__error">
          <strong>Error:</strong> {error}
        </div>
      )}

      <div className={`live__feed-container ${running ? "" : "live__feed-container--hidden"}`}>
        <div className="live__feed">
          <video
            ref={videoRef}
            muted
            playsInline
            className="live__video"
          />
          <span className="live__feed-label">Raw Feed</span>
        </div>
        {annotatedSrc && (
          <div className="live__feed">
            <img
              src={annotatedSrc}
              alt="Annotated live feed"
              className="live__annotated"
            />
            <span className="live__feed-label">Model Output</span>
          </div>
        )}
      </div>

      {gazeData && (
        <div className="live__stats">
          <div className="live__stat">
            <span className="live__stat-label">Gaze Pitch</span>
            <span className="live__stat-value">
              {gazeData.pitch_deg?.toFixed(1)}°
            </span>
          </div>
          <div className="live__stat">
            <span className="live__stat-label">Gaze Yaw</span>
            <span className="live__stat-value">
              {gazeData.yaw_deg?.toFixed(1)}°
            </span>
          </div>
          <div className="live__stat">
            <span className="live__stat-label">Head Pitch</span>
            <span className="live__stat-value">
              {gazeData.head_pitch_deg?.toFixed(1)}°
            </span>
          </div>
          <div className="live__stat">
            <span className="live__stat-label">Head Yaw</span>
            <span className="live__stat-value">
              {gazeData.head_yaw_deg?.toFixed(1)}°
            </span>
          </div>
          {gazeData.gaze_vector && (
            <div className="live__stat live__stat--wide">
              <span className="live__stat-label">3D Vector</span>
              <code className="live__stat-value">
                [{gazeData.gaze_vector.map((v) => v?.toFixed(3)).join(", ")}]
              </code>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
