import torch
import torch.nn as nn
import numpy as np


def pitchyaw_to_vector(pitchyaw):
    pitch = pitchyaw[:, 0]
    yaw   = pitchyaw[:, 1]
    x = -torch.cos(pitch) * torch.sin(yaw)
    y = -torch.sin(pitch)
    z = -torch.cos(pitch) * torch.cos(yaw)
    return torch.stack([x, y, z], dim=1)


def vector_to_pitchyaw(vectors):
    x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    norm = torch.sqrt(x*x + y*y + z*z).clamp(min=1e-7)
    x, y, z = x / norm, y / norm, z / norm
    pitch = torch.asin((-y).clamp(-1 + 1e-7, 1 - 1e-7))
    yaw   = torch.atan2(-x, -z)
    return torch.stack([pitch, yaw], dim=1)


def compute_angular_error(pred_py, true_py):
    with torch.no_grad():
        pred_v = pitchyaw_to_vector(pred_py.float())
        true_v = pitchyaw_to_vector(true_py.float())
        cosine = torch.sum(pred_v * true_v, dim=1)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        degrees = torch.rad2deg(torch.acos(cosine))
    return degrees.mean().item()


class GazeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._l1 = nn.L1Loss()

    def forward(self, pred, target):
        return self._l1(pred, target)


class AverageMeter:
    def __init__(self, name=''):
        self.name = name
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / max(self.count, 1)

    def __repr__(self):
        return f'{self.name}: {self.avg:.4f}'
