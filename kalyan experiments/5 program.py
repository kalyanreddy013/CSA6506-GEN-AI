import numpy as np
import pandas as pd

print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)

try:
    import sklearn
    print("Scikit-learn version:", sklearn.__version__)
except ImportError:
    print("Scikit-learn is not installed")

try:
    import tensorflow as tf
    print("TensorFlow version:", tf.__version__)
except ImportError:
    print("TensorFlow is not installed")

try:
    import torch
    print("PyTorch version:", torch.__version__)
except ImportError:
    print("PyTorch is not installed")

try:
    import cv2
    print("OpenCV version:", cv2.__version__)
except ImportError:
    print("OpenCV is not installed")