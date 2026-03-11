import os
import cv2
import numpy as np
import torch
import mediapipe as mp
import torchvision.transforms as T
from PIL import Image

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from core.model import GazeSymCAT
from core.utils import pitchyaw_to_vector
from core.dataset import EyeExtractor

MODEL_PATH   = os.path.join(os.path.dirname(__file__), "weights", "best_model.pth")
CAMERA_INDEX = 0

_IS_CPU = not torch.cuda.is_available() or os.environ.get('CUDA_VISIBLE_DEVICES') == ''

if _IS_CPU:
    CAPTURE_W    = 320
    CAPTURE_H    = 240
    DISPLAY_W    = 640
    DISPLAY_H    = 480
    print("[live_test] CPU mode detected - using lower resolution for speed")
else:
    CAPTURE_W    = 640
    CAPTURE_H    = 480
    DISPLAY_W    = 900
    DISPLAY_H    = 700
    print("[live_test] GPU mode detected - using full resolution")

ARROW_SCALE  = 0.35
SAVE_DIR     = "live_out"

_FACE_MESH = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

_MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),
    (0.0,  -330.0,  -65.0),
    (-225.0, 170.0, -135.0),
    (225.0,  170.0, -135.0),
    (-150.0,-150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)
_LM_INDICES = [1, 152, 263, 33, 287, 57]


def get_head_pose(pil_image: Image.Image) -> np.ndarray:
    image = np.array(pil_image.convert("RGB"))
    h, w  = image.shape[:2]

    results = _FACE_MESH.process(image)
    if not results.multi_face_landmarks:
        return np.zeros(2, dtype=np.float32)

    lm = results.multi_face_landmarks[0].landmark
    image_points = np.array(
        [(lm[i].x * w, lm[i].y * h) for i in _LM_INDICES], dtype=np.float64)

    focal  = w
    cam_mx = np.array([[focal, 0, w/2],
                        [0, focal, h/2],
                        [0,     0,   1]], dtype=np.float64)

    _, rvec, _ = cv2.solvePnP(
        _MODEL_POINTS, image_points, cam_mx,
        np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)

    rmat, _ = cv2.Rodrigues(rvec)
    pitch   = np.arcsin(-rmat[2, 1])
    yaw     = np.arctan2(rmat[2, 0], rmat[2, 2])
    return np.array([pitch, yaw], dtype=np.float32)


_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def preprocess(pil_img: Image.Image, device: torch.device):
    extractor = EyeExtractor()
    face_crop, leye, reye, face_center = extractor.extract(pil_img)
    face_t = _TRANSFORM(face_crop).unsqueeze(0).to(device)
    leye_t = _TRANSFORM(leye).unsqueeze(0).to(device)
    reye_t = _TRANSFORM(reye).unsqueeze(0).to(device)
    return face_t, leye_t, reye_t, leye, reye, face_center


def draw_annotations(frame_bgr: np.ndarray,
                     gaze_vec:   np.ndarray,
                     head_pose:  np.ndarray,
                     leye_pil:   Image.Image,
                     reye_pil:   Image.Image,
                     fps:        float,
                     face_center: tuple = None) -> np.ndarray:
    h, w = frame_bgr.shape[:2]

    if face_center:
        cx, cy = face_center
    else:
        cx, cy = w // 2, h // 2

    arrow_len = int(min(w, h) * ARROW_SCALE)

    gx = int(cx + gaze_vec[0] * arrow_len)
    gy = int(cy + gaze_vec[1] * arrow_len)
    cv2.arrowedLine(frame_bgr, (cx, cy), (gx, gy),
                    (0, 255, 0), 4, tipLength=0.25)
    cv2.putText(frame_bgr, "Gaze", (gx + 8, gy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    hx = int(cx + np.sin(head_pose[1]) * arrow_len)
    hy = int(cy - np.sin(head_pose[0]) * arrow_len)
    cv2.arrowedLine(frame_bgr, (cx, cy), (hx, hy),
                    (255, 100, 0), 3, tipLength=0.25)
    cv2.putText(frame_bgr, "Head", (hx + 8, hy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)

    eye_sz = 120
    for idx, eye_pil in enumerate([leye_pil, reye_pil]):
        eye_bgr = cv2.cvtColor(
            np.array(eye_pil.resize((eye_sz, eye_sz))), cv2.COLOR_RGB2BGR)
        x0 = idx * (eye_sz + 6)
        y0 = h - eye_sz - 6
        frame_bgr[y0:y0 + eye_sz, x0:x0 + eye_sz] = eye_bgr
    cv2.putText(frame_bgr, "L", (6, h - eye_sz - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame_bgr, "R", (eye_sz + 14, h - eye_sz - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    pitch_deg = float(np.degrees(
        np.arctan2(-gaze_vec[1],
                   np.sqrt(gaze_vec[0]**2 + gaze_vec[2]**2))))
    yaw_deg   = float(np.degrees(np.arctan2(-gaze_vec[0], -gaze_vec[2])))

    cv2.putText(frame_bgr, f"Pitch: {pitch_deg:+.1f} deg", (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame_bgr, f"Yaw:   {yaw_deg:+.1f} deg",  (10, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame_bgr,
                f"HP  Pitch: {np.degrees(head_pose[0]):+.1f}  "
                f"Yaw: {np.degrees(head_pose[1]):+.1f} deg",
                (10, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 100, 0), 2)
    cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (w - 130, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
    cv2.putText(frame_bgr, "Q: quit   S: snapshot", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    return frame_bgr


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[live_test] Running on {device}")

    model = GazeSymCAT(d_model=512, num_blocks=2, use_head_pose=True).to(device)
    if os.path.exists(MODEL_PATH):
        print(f"[live_test] Loading weights from {MODEL_PATH}")
        ckpt = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(
            ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    else:
        print("[live_test] WARNING: No checkpoint found — using random weights.")
    model.eval()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
    print(f"[live_test] Camera {CAMERA_INDEX} opened at {CAPTURE_W}x{CAPTURE_H}.  Press Q to quit, S to snapshot.")

    os.makedirs(SAVE_DIR, exist_ok=True)
    snapshot_count = 0
    tick = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[live_test] Failed to grab frame.")
            break

        now   = cv2.getTickCount()
        fps   = cv2.getTickFrequency() / (now - tick)
        tick  = now

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        try:
            face_t, leye_t, reye_t, leye_pil, reye_pil, face_center = preprocess(pil_img, device)
            head_pose_np = get_head_pose(pil_img)
            head_pose_t  = torch.tensor(head_pose_np).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_py = model(face_t, leye_t, reye_t, head_pose=head_pose_t)

            gaze_vec = pitchyaw_to_vector(pred_py)[0].cpu().numpy()

        except Exception as e:
            cv2.putText(frame, f"Detection failed: {e}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Live Gaze Estimation", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        display = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))

        scale_x = DISPLAY_W / frame.shape[1]
        scale_y = DISPLAY_H / frame.shape[0]
        display_face_center = (int(face_center[0] * scale_x), int(face_center[1] * scale_y))

        display = draw_annotations(display, gaze_vec, head_pose_np,
                                   leye_pil, reye_pil, fps, display_face_center)

        cv2.imshow("Live Gaze Estimation", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            snap_path = os.path.join(SAVE_DIR, f"snapshot_{snapshot_count:04d}.jpg")
            cv2.imwrite(snap_path, display)
            print(f"[live_test] Snapshot saved → {snap_path}")
            snapshot_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("[live_test] Done.")


if __name__ == "__main__":
    main()
