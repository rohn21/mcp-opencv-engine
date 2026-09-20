import cv2
import numpy as np

def apply_grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def apply_edges(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 100, 200)

def apply_blur(img: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(img, (15, 15), 0)

def apply_threshold(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return thresh

FILTER_REGISTRY = {
    "grayscale": apply_grayscale,
    "edges":     apply_edges,
    "blur":      apply_blur,
    "threshold": apply_threshold,
}
