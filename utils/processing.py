import cv2
import numpy as np
import math
import pytesseract

from utils.image_utils import StepLogger


# =========================
# Metrics
# =========================

def mse(image1, image2):
    return np.mean((image1.astype("float") - image2.astype("float")) ** 2)


def psnr(image1, image2):
    mse_value = mse(image1, image2)

    if mse_value == 0:
        return float("inf")

    return 20 * math.log10(255.0 / math.sqrt(mse_value))

# =========================
# Point Ordering
# =========================

def order_points(pts):
    pts = np.array(pts).reshape((4, 2))

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

# =========================
# Distance
# =========================

def euclidean(a, b):
    return np.sqrt(((a[0] - b[0]) ** 2) + ((b[1] - a[1]) ** 2))


# =========================
# Main Pipeline
# =========================

def process_document(image):

    logger = StepLogger()

    # =========================
    # Original
    # =========================

    original = image.copy()
    logger.add_step("Original Image", original)

    # =========================
    # Gray Scale
    # =========================

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    logger.add_step("Gray Scale", gray)

    # =========================
    # Filters
    # =========================

    gaussian = cv2.GaussianBlur(gray, (7, 7), 0)
    logger.add_step("Gaussian Filter", gaussian)

    median = cv2.medianBlur(gray, 7)
    logger.add_step("Median Filter", median)

    fast_nl = cv2.fastNlMeansDenoising(gray)
    logger.add_step("FastNL Means", fast_nl)

    # =========================
    # Filter Comparison
    # =========================

    gaussian_psnr = psnr(gray, gaussian)
    median_psnr = psnr(gray, median)
    fastnl_psnr = psnr(gray, fast_nl)

    filters = {
        "gaussian": (gaussian, gaussian_psnr),
        "median": (median, median_psnr),
        "fastnl": (fast_nl, fastnl_psnr),
    }

    best_filter_name = max(filters, key=lambda x: filters[x][1])
    best_filtered = filters[best_filter_name][0]

    logger.add_step(f"Best Filter ({best_filter_name})", best_filtered)

    # =========================
    # Threshold
    # =========================

    _, thresh = cv2.threshold(
        best_filtered,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    logger.add_step("Threshold", thresh)

    # =========================
    # Morphology
    # =========================

    kernel_size = (2, 2)

    square_kernel = np.ones(kernel_size, np.uint8)

    morph = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        square_kernel,
        iterations=3,
    )

    logger.add_step("Morphology", morph)

    # =========================
    # Find Contours
    # =========================

    contours, _ = cv2.findContours(
        morph,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contour_image = np.zeros_like(morph)

    contour_image = cv2.drawContours(
        contour_image,
        contours,
        -1,
        (255, 255, 255),
        2,
    )

    logger.add_step("All Contours", contour_image)

    # =========================
    # Sort Contours
    # =========================

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    candidates = []

    for index, contour in enumerate(contours[:10]):

        peri = cv2.arcLength(contour, True)

        approx = cv2.approxPolyDP(
            contour,
            0.025 * peri,
            True,
        )

        blank = np.zeros_like(morph)

        attempt = cv2.drawContours(
            blank,
            [approx],
            -1,
            (255, 255, 255),
            2,
        )

        logger.add_step(f"Contour Approximation {index + 1}", attempt)

        if len(approx) == 4:
            candidates.append((cv2.contourArea(contour), approx))

    # =========================
    # Choose Document Contour
    # =========================

    if len(candidates) == 0:
        return {
            "success": False,
            "steps": logger.get_steps(),
            "message": "No 4-corner document detected"
        }

    candidates.sort(key=lambda x: x[0], reverse=True)

    doc_contour = candidates[0][1]

    contour_result = original.copy()

    contour_result = cv2.drawContours(
        contour_result,
        [doc_contour],
        -1,
        (0, 255, 0),
        5,
    )

    for point in doc_contour:
        x, y = point[0]
        cv2.circle(contour_result, (x, y), 10, (255, 0, 0), -1)

    logger.add_step("Detected Document", contour_result)

    # =========================
    # Perspective Transform
    # =========================

    rect = order_points(doc_contour)

    (tl, tr, br, bl) = rect

    widthA = euclidean(br, bl)
    widthB = euclidean(tr, tl)

    maxWidth = max(int(widthA), int(widthB))

    heightA = euclidean(tr, br)
    heightB = euclidean(tl, bl)

    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)

    warped = cv2.warpPerspective(
        original,
        matrix,
        (maxWidth, maxHeight),
    )

    logger.add_step("Perspective Transform", warped)

    # =========================
    # OCR Preparation
    # =========================

    transformed_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    logger.add_step("Transformed Gray", transformed_gray)

    # =========================
    # OCR
    # =========================

    extracted_text = pytesseract.image_to_string(transformed_gray)

    return {
        "success": True,
        "steps": logger.get_steps(),
        "text": extracted_text,
        "final_image": transformed_gray,
    }