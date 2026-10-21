%%writefile opencv_basics_demo.py
import cv2
import numpy as np

import config


def make_demo_image(width=640, height=480):
    img = np.full((height, width, 3), 245, dtype=np.uint8)

    cv2.rectangle(img, (60, 60), (220, 200), (40, 90, 200), thickness=-1)
    cv2.circle(img, (420, 140), 90, (90, 170, 60), thickness=-1)
    cv2.ellipse(img, (250, 350), (120, 60), 25, 0, 360, (200, 120, 40), thickness=-1)
    triangle = np.array([[450, 260], [560, 420], [370, 420]], dtype=np.int32)
    cv2.fillPoly(img, [triangle], (60, 60, 180))

    noise = np.random.default_rng(0).normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def process(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.2)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated = img.copy()
    shape_info = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 500:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.drawContours(annotated, [c], -1, (0, 255, 0), 2)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 1)
        shape_info.append({"area": float(area), "bbox": (x, y, w, h)})

    return gray, blurred, edges, annotated, shape_info


def camera_calibration_snippet():
    return """
# Standard OpenCV camera calibration flow (reference -- needs real checkerboard images):
#
# checkerboard_size = (9, 6)  # inner corners
# objpoints, imgpoints = [], []
# objp = np.zeros((checkerboard_size[0]*checkerboard_size[1], 3), np.float32)
# objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
#
# for fname in calibration_images:
#     img = cv2.imread(fname)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     found, corners = cv2.findChessboardCorners(gray, checkerboard_size)
#     if found:
#         objpoints.append(objp)
#         imgpoints.append(corners)
#
# ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
#     objpoints, imgpoints, gray.shape[::-1], None, None
# )
# undistorted = cv2.undistort(img, camera_matrix, dist_coeffs)
"""


def make_panel(images_with_titles):
    cell_h, cell_w = 240, 320
    cols = 3
    rows = (len(images_with_titles) + cols - 1) // cols
    panel = np.full((rows * cell_h, cols * cell_w, 3), 30, dtype=np.uint8)

    for idx, (title, im) in enumerate(images_with_titles):
        r, c = divmod(idx, cols)
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        resized = cv2.resize(im, (cell_w - 10, cell_h - 30))
        y0, x0 = r * cell_h + 25, c * cell_w + 5
        panel[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
        cv2.putText(panel, title, (c * cell_w + 8, r * cell_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def main():
    img = make_demo_image()
    cv2.imwrite(config.DEMO_IMAGE_PATH, img)

    gray, blurred, edges, annotated, shape_info = process(img)

    print(f"Found {len(shape_info)} shapes:")
    for s in shape_info:
        print(f"  area={s['area']:.0f}  bbox={s['bbox']}")

    panel = make_panel([
        ("original", img),
        ("grayscale", gray),
        ("gaussian blur", blurred),
        ("canny edges", edges),
        ("contours + bboxes", annotated),
    ])
    cv2.imwrite(config.DEMO_OUTPUT_PATH, panel)
    print(f"\nSaved pipeline panel to {config.DEMO_OUTPUT_PATH}")
    print(camera_calibration_snippet())


if __name__ == "__main__":
    main()
