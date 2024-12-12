import numpy as np
from scipy.ndimage import zoom

def normalize_scan(scan):
    """标准化扫描数据"""
    scan = (scan - scan.mean()) / (scan.std() + 1e-8)
    return scan

def resize_volume(img, target_shape=(128, 128, 128)):
    """调整3D体积大小"""
    current_shape = img.shape
    factors = (
        target_shape[0] / current_shape[0],
        target_shape[1] / current_shape[1],
        target_shape[2] / current_shape[2]
    )
    return zoom(img, factors, order=1) 