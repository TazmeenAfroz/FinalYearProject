import os
import math
import warnings
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.model import GazeSymCAT
from core.dataset import EyeExtractor
from core.utils import pitchyaw_to_vector

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "weights")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AVAILABLE_MODELS = {
    "best_model": "best_model.pth",
    "best_modelFull": "best_modelFull.pth",
}
DEFAULT_MODEL = "best_model"

_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

_FACE_MESH = None
_MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),
    (0.0,  -330.0,  -65.0),
    (-225.0, 170.0, -135.0),
    (225.0,  170.0, -135.0),
    (-150.0,-150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)
_LM_INDICES = [1, 152, 263, 33, 287, 57]


def _get_face_mesh():
    global _FACE_MESH
    if _FACE_MESH is None:
        import mediapipe as mp
        _FACE_MESH = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.2,
            model_complexity=1)
    return _FACE_MESH


def get_head_pose(pil_image: Image.Image) -> np.ndarray:
    import cv2
    image = np.array(pil_image.convert("RGB"))
    h, w = image.shape[:2]
    try:
        results = _get_face_mesh().process(image)
    except Exception:
        return np.zeros(2, dtype=np.float32)
    if not results.multi_face_landmarks:
        return np.zeros(2, dtype=np.float32)
    lm = results.multi_face_landmarks[0].landmark
    image_points = np.array(
        [(lm[i].x * w, lm[i].y * h) for i in _LM_INDICES], dtype=np.float64)
    focal = w
    cam_mx = np.array([[focal, 0, w / 2],
                        [0, focal, h / 2],
                        [0,     0,    1]], dtype=np.float64)
    _, rvec, _ = cv2.solvePnP(
        _MODEL_POINTS, image_points, cam_mx,
        np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    rmat, _ = cv2.Rodrigues(rvec)
    pitch = float(np.arcsin(-rmat[2, 1]))
    yaw = float(np.arctan2(rmat[2, 0], rmat[2, 2]))
    return np.array([pitch, yaw], dtype=np.float32)


class _ModelRegistry:
    def __init__(self):
        self._models = {}
        self._extractor = None

    def get(self, model_name: str = None):
        name = model_name or DEFAULT_MODEL
        if name not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model '{name}'. Available: {list(AVAILABLE_MODELS.keys())}")
        if name not in self._models:
            self._load(name)
        if self._extractor is None:
            self._extractor = EyeExtractor(output_size=224)
        return self._models[name], self._extractor

    def _load(self, name: str):
        path = os.path.join(MODEL_DIR, AVAILABLE_MODELS[name])
        print(f"[inference] Loading GazeSymCAT model '{name}' …")
        use_hp = True
        model = GazeSymCAT(d_model=512, num_blocks=2, use_head_pose=use_hp)
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
            sd = ckpt.get("model_state_dict", ckpt)
            model_sd = model.state_dict()
            filtered = {k: v for k, v in sd.items()
                        if k in model_sd and model_sd[k].shape == v.shape}
            skipped = [k for k, v in sd.items()
                       if k in model_sd and model_sd[k].shape != v.shape]
            if skipped:
                print(f"[inference] Skipped {len(skipped)} shape-mismatched tensors")
            model.load_state_dict(filtered, strict=False)
            print(f"[inference] Loaded {len(filtered)}/{len(model_sd)} weights from checkpoint")
        else:
            print(f"[inference] WARNING: no checkpoint found for '{name}' — random weights")
        model.to(DEVICE).eval()
        self._models[name] = model
        print(f"[inference] '{name}' ready on {DEVICE}")


_registry = _ModelRegistry()


def get_available_models():
    return {"models": list(AVAILABLE_MODELS.keys()), "default": DEFAULT_MODEL}


def predict(pil_image: Image.Image, model_name: str = None):
    model, extractor = _registry.get(model_name)

    pil_rgb = pil_image.convert("RGB")

    try:
        face_crop, leye, reye, face_center = extractor.extract(pil_rgb)
    except Exception as e:
        raise ValueError(f"Could not detect face/eyes in the image: {e}")

    face_t = _TRANSFORM(face_crop).unsqueeze(0).to(DEVICE)
    leye_t = _TRANSFORM(leye).unsqueeze(0).to(DEVICE)
    reye_t = _TRANSFORM(reye).unsqueeze(0).to(DEVICE)

    head_pose_np = get_head_pose(pil_rgb)
    head_pose_t = torch.tensor(head_pose_np).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_py = model(face_t, leye_t, reye_t, head_pose=head_pose_t)

    pitch_rad = float(pred_py[0, 0].cpu())
    yaw_rad = float(pred_py[0, 1].cpu())

    gaze_vec = pitchyaw_to_vector(pred_py)[0].cpu().numpy().tolist()

    annotated = _draw_gaze(pil_rgb.copy(), pitch_rad, yaw_rad, gaze_vec,
                           head_pose_np, face_center)

    return {
        "pitch_rad": pitch_rad,
        "yaw_rad": yaw_rad,
        "pitch_deg": math.degrees(pitch_rad),
        "yaw_deg": math.degrees(yaw_rad),
        "gaze_vector": gaze_vec,
        "head_pitch_deg": float(np.degrees(head_pose_np[0])),
        "head_yaw_deg": float(np.degrees(head_pose_np[1])),
        "annotated_image": annotated,
    }


def _draw_gaze(pil_img: Image.Image, pitch: float, yaw: float,
               gaze_vec: list, head_pose: np.ndarray, face_center: tuple) -> Image.Image:
    import cv2

    img = np.array(pil_img)
    h, w = img.shape[:2]

    cx, cy = face_center
    arrow_len = int(min(w, h) * 0.35)

    gx = int(cx + gaze_vec[0] * arrow_len)
    gy = int(cy + gaze_vec[1] * arrow_len)
    cv2.arrowedLine(img, (cx, cy), (gx, gy), (0, 255, 0), 3, tipLength=0.25)

    hx = int(cx + np.sin(head_pose[1]) * arrow_len)
    hy = int(cy - np.sin(head_pose[0]) * arrow_len)
    cv2.arrowedLine(img, (cx, cy), (hx, hy), (255, 100, 0), 2, tipLength=0.25)

    cv2.putText(img, f"Gaze  P:{math.degrees(pitch):+.1f} Y:{math.degrees(yaw):+.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img, f"Head  P:{np.degrees(head_pose[0]):+.1f} Y:{np.degrees(head_pose[1]):+.1f}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
    cv2.putText(img, "Gaze", (gx + 8, gy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img, "Head", (hx + 8, hy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

    return Image.fromarray(img)
