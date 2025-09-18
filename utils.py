import cv2
import numpy as np

def pixel_to_cartesian(u, v, W, H, system="centered"):
    """
    Convert image pixel coordinates (u,v) to 2D Cartesian coordinates.

    Parameters
    ----------
    u, v : int or float
        Pixel coordinates (u = column, v = row).
    W, H : int
        Width and height of the image.
    system : str
        "topleft"  -> origin at top-left, x right, y down.
        "bottomleft" -> origin at bottom-left, x right, y up.
        "centered" -> origin at image center, x right, y up.

    Returns
    -------
    x, y : float
        Cartesian coordinates.
    """
    if system == "topleft":
        x = float(u)
        y = float(v)
    elif system == "bottomleft":
        x = float(u)
        y = float(H - 1 - v)
    elif system == "centered":
        x = float(u) - (W - 1) / 2.0
        y = (H - 1) / 2.0 - float(v)
    else:
        raise ValueError("system must be 'topleft', 'bottomleft', or 'centered'")
    return x, y

def play_video(path, as_gray=True, scale=1.0, delay_ms=None,
               apply_threshold=False, thresh=200, show_mask=True):
    """
    Play a video, optionally grayscale, with optional brightness thresholding.

    Controls:
      q = quit
      space = pause/resume
      n = next frame (when paused)
      r = restart
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0
    base_delay = int(max(1, round(1000.0 / fps)))
    delay = base_delay if delay_ms is None else int(delay_ms)

    cv2.namedWindow("video", cv2.WINDOW_NORMAL)
    if apply_threshold and show_mask:
        cv2.namedWindow("mask", cv2.WINDOW_NORMAL)

    paused = False
    restart = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            if as_gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if apply_threshold:
                # ensure uint8 grayscale
                if frame.ndim != 2:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame
                if gray.dtype != np.uint8:
                    gray = cv2.normalize(gray.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                _, mask = cv2.threshold(gray, int(thresh), 255, cv2.THRESH_BINARY)
                display_frame = cv2.bitwise_and(gray, gray, mask=mask)
            else:
                display_frame = frame

            if scale != 1.0:
                h, w = display_frame.shape[:2]
                display_frame = cv2.resize(display_frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
                if apply_threshold and show_mask:
                    mask = cv2.resize(mask, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_NEAREST)

            cv2.imshow("video", display_frame)
            if apply_threshold and show_mask:
                cv2.imshow("mask", mask)

        key = cv2.waitKey(0 if paused else delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('n'):
            paused = True
            continue
        elif key == ord('r'):
            restart = True
            break

    cap.release()
    if apply_threshold and show_mask:
        cv2.destroyWindow("mask")
    cv2.destroyWindow("video")

    if restart:
        play_video(path, as_gray=as_gray, scale=scale, delay_ms=delay if delay_ms is not None else None,
                   apply_threshold=apply_threshold, thresh=thresh, show_mask=show_mask)