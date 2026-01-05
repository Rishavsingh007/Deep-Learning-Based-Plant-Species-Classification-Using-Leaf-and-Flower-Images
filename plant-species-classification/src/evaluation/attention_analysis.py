"""
CT7160NI Computer Vision Coursework
Attention Mechanism Analysis

This module provides utilities for analyzing Grad-CAM attention maps.
"""

import torch
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import os


def threshold_attention_map(attention_map, threshold=0.5):
    """
    Threshold attention map to create binary mask.
    
    Parameters:
    -----------
    attention_map : numpy.ndarray
        Grad-CAM attention map (normalized 0-1)
    threshold : float
        Threshold value (0-1)
        
    Returns:
    --------
    numpy.ndarray : Binary mask
    """
    binary = (attention_map >= threshold).astype(np.uint8) * 255
    return binary



