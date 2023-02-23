import numpy as np
import cv2


def removeLines(result, axis) -> np.ndarray:
    img = result.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    if axis == "horizontal":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    elif axis == "vertical":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    else:
        raise ValueError("Axis must be either 'horizontal' or 'vertical'")
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    result = img.copy()
    for c in cnts:
        cv2.drawContours(result, [c], -1, (255, 255, 255), 2)
    return result
