import os
import torch
import numpy as np

PROJECT_ROOT = r"D:\WorkSpace\pycharm\Python学习路线\recommender_project"
src = os.path.join(PROJECT_ROOT, "hybrid_ml1m_best.pth")
dst = os.path.join(PROJECT_ROOT, "hybrid_ml1m_best_safe.pth")

ckpt = torch.load(src, map_location="cpu", weights_only=False)

def to_py(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(x) for x in obj]
    return obj

safe_ckpt = {}
for k, v in ckpt.items():
    if k in ["idx2user", "idx2item", "user2idx", "item2idx", "config"]:
        safe_ckpt[k] = to_py(v)
    else:
        safe_ckpt[k] = v

torch.save(safe_ckpt, dst)
print("Saved:", dst)
