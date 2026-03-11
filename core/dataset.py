import os
import glob
import random
import warnings

warnings.filterwarnings('ignore', category=UserWarning,
                        message='.*SymbolDatabase.GetPrototype.*')
warnings.filterwarnings('ignore', category=UserWarning,
                        module='google.protobuf.*')

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


class EyeExtractor:
    _L_IDXS = [263,249,390,373,374,380,381,382,362,466,388,387,386,385,384,398]
    _R_IDXS = [33,7,163,144,145,153,154,155,133,246,161,160,159,158,157,173]

    def __init__(self, output_size=224):
        self.output_size = output_size
        self._detector   = None
        self._use_fallback_only = False

    def _init_detector(self):
        if self._detector is not None or self._use_fallback_only:
            return

        try:
            import mediapipe as mp
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self._detector = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True, max_num_faces=1,
                    refine_landmarks=True, min_detection_confidence=0.3)
            print('[EyeExtractor] Using mediapipe FaceMesh (legacy API).')
        except Exception as e:
            print(f'[EyeExtractor] MediaPipe init failed: {e}. Using fallback for ALL images.')
            self._use_fallback_only = True
            self._detector = None

    def extract(self, img_pil, test_mode=False):
        if test_mode:
            self._init_detector()
            if self._detector is None:
                self._use_fallback_only = True
                return self._fallback_gazesetmerge(img_pil)

        if self._use_fallback_only:
            return self._fallback_gazesetmerge(img_pil)

        self._init_detector()
        if self._detector is None:
            return self._fallback_gazesetmerge(img_pil)

        img_np = np.array(img_pil, dtype=np.uint8)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result = self._detector.process(img_np)

            if not result.multi_face_landmarks:
                return self._fallback_gazesetmerge(img_pil)

            lm   = result.multi_face_landmarks[0].landmark
            h, w = img_np.shape[:2]
            pts  = np.array([(l.x * w, l.y * h) for l in lm])

            face_cx, face_cy = pts.mean(axis=0)

            face_crop = self._crop_region(img_pil, pts, padding_multiplier=1.5)
            leye = self._crop_region(img_pil, pts[self._L_IDXS])
            reye = self._crop_region(img_pil, pts[self._R_IDXS])
            return face_crop, leye, reye, (int(face_cx), int(face_cy))

        except Exception as e:
            if test_mode:
                print(f'[EyeExtractor] MediaPipe processing failed on first image: {e}. '
                      f'Using fallback for ALL images.')
                self._use_fallback_only = True
            return self._fallback_gazesetmerge(img_pil)

    def _crop_region(self, img_pil, pts, padding_multiplier=1.5):
        x_min, y_min = pts.min(0)
        x_max, y_max = pts.max(0)
        cx, cy       = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
        size         = max(x_max - x_min, y_max - y_min) * padding_multiplier
        s2           = size / 2.0
        iw, ih       = img_pil.size
        box = (
            max(0.0, cx - s2),
            max(0.0, cy - s2),
            min(float(iw), cx + s2),
            min(float(ih), cy + s2),
        )
        return img_pil.crop(box).resize((self.output_size, self.output_size))

    def _fallback_gazesetmerge(self, img_pil):
        w, h = img_pil.size

        eye_band_top    = int(h * 0.15)
        eye_band_bottom = int(h * 0.50)
        eye_band_h      = eye_band_bottom - eye_band_top

        reye_box = (
            0,
            eye_band_top,
            w // 2,
            eye_band_bottom,
        )

        leye_box = (
            w // 2,
            eye_band_top,
            w,
            eye_band_bottom,
        )

        face_margin = int(w * 0.1)
        face_box = (
            face_margin,
            0,
            w - face_margin,
            int(h * 0.85)
        )

        face_center = (w // 2, h // 2)

        face = img_pil.crop(face_box).resize((self.output_size, self.output_size))
        leye = img_pil.crop(leye_box).resize((self.output_size, self.output_size))
        reye = img_pil.crop(reye_box).resize((self.output_size, self.output_size))
        return face, leye, reye, face_center


class MultiH5Dataset(Dataset):
    def __init__(self, data_dir, transform=None, augment=False):
        self.transform = transform
        self.augment   = augment
        self.extractor = EyeExtractor(output_size=224)
        self._first_image_tested = False

        h5_files = sorted(glob.glob(os.path.join(data_dir, '*.h5')))
        self.index_map = []
        for fpath in h5_files:
            try:
                with h5py.File(fpath, 'r') as f:
                    if 'face_patch' not in f:
                        print(f'[WARN] {os.path.basename(fpath)}: no face_patch key, skipping')
                        continue
                    n = f['face_patch'].shape[0]
                    self.index_map.extend((fpath, i) for i in range(n))
            except Exception as e:
                print(f'[WARN] Cannot open {os.path.basename(fpath)}: {e}')

        print(f'[Dataset] {data_dir}: {len(h5_files)} files, '
              f'{len(self.index_map):,} samples (augment={augment})')

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx, _start=None):
        if _start is None:
            _start = idx

        fpath, local_i = self.index_map[idx]
        try:
            with h5py.File(fpath, 'r') as f:
                face_bgr = f['face_patch'][local_i]
                gaze_np  = f['face_gaze'][local_i].astype(np.float32)
                if 'face_head_pose' in f:
                    hp = f['face_head_pose'][local_i].astype(np.float32)[:2]
                else:
                    hp = np.zeros(2, dtype=np.float32)

            face_rgb = face_bgr[..., ::-1].copy()
            face_pil = Image.fromarray(face_rgb.astype(np.uint8))

            test_mode = not self._first_image_tested
            if test_mode:
                self._first_image_tested = True

            _, leye_pil, reye_pil, _ = self.extractor.extract(face_pil, test_mode=test_mode)

            label = gaze_np.copy()
            if self.augment and random.random() > 0.5:
                face_pil         = TF.hflip(face_pil)
                leye_pil         = TF.hflip(leye_pil)
                reye_pil         = TF.hflip(reye_pil)
                leye_pil, reye_pil = reye_pil, leye_pil
                label[1]         = -label[1]
                hp[1]            = -hp[1]

            if self.transform:
                face = self.transform(face_pil)
                leye = self.transform(leye_pil)
                reye = self.transform(reye_pil)
            else:
                import torchvision.transforms as T
                _t = T.Compose([T.Resize((224, 224)), T.ToTensor()])
                face = _t(face_pil)
                leye = _t(leye_pil)
                reye = _t(reye_pil)

            label     = torch.tensor(label, dtype=torch.float32)
            head_pose = torch.tensor(hp,    dtype=torch.float32)
            return face, leye, reye, label, head_pose

        except Exception as e:
            next_idx = (idx + 1) % len(self)
            if next_idx == _start:
                raise RuntimeError(f'All samples corrupted in dataset! Last error: {e}')
            return self.__getitem__(next_idx, _start=_start)
