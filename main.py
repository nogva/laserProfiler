from utils import *
from dotRocognition import *
from pixelToCartesian import *
import cv2
import matplotlib.pyplot as plt

#config:
play_video = True

def main():
    # play_video("video\initial_test.mp4", as_gray=True, scale=1, delay_ms=None, apply_threshold=True, thresh=249, show_mask=False)
    video_path = r"video\initial_run3.mp4"
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    assert cap.isOpened(), "Cannot open video"

    # Read properties with fallbacks
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1e-3 else 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    assert width > 0 and height > 0, "Invalid video dimensions"

    acc = CartesianAccumulator((height, width))  # global union/counts over all frames
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # main loop for video processing
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot fetch the frame.")
            break

        # Process the frame
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(grey, 249, 255, cv2.THRESH_BINARY)
        # mask must be uint8 with values {0,255}
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        if play_video:
            # show the frames
            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("frame", mask)
            delay = int(1000 / fps)  # milliseconds
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

        # ---- Per-frame Cartesian points ----
        # Since mask is 0/255, pass thresh=None to include all nonzero pixels.
        pts = img_to_cartesian(
            img=mask,
            thresh=None,   # or thresh=255 to include exactly 255-valued pixels
            invert=False,
            pixel_pitch=1.0,  # set to your physical scale if needed
            centered=True     # False → origin at top-left pixel (0,0)
        )
        # do math on pts here (centroid, PCA, fitting, etc.)

        # ---- Accumulate across frames (union + counts) ----
        acc.update(mask, thresh=None)  # uses same foreground rule

    # Release resources after loop
    cap.release()
    cv2.destroyAllWindows()

    # All frames processed; acc is your CartesianAccumulator(...)
    acc.identify_bounds()
    acc.identify_center()
    acc.remove_unwanted_around_center(radius_to_width_ratio= 0.25)

    acc.point_metrics(should_print=True, scale=40/341.84)

    # plotting
    fig, ax = plt.subplots(figsize=(6, 6))

    # use the cached points you stored as acc.points (shape: N x 2)
    pts = acc.pts
    if pts is None or pts.size == 0:
        raise ValueError("acc.points is empty. Run update_points(...) first.")

    ax.scatter(pts[:, 0], pts[:, 1], s=10, marker='.', linewidths=0)

    # optional: draw bounds if available (expects list of (x, y) corners, Cartesian order)
    if acc.bounds:
        xs, ys = zip(*(acc.bounds + [acc.bounds[0]]))  # close loop
        ax.plot(xs, ys, color="red", linewidth=2)

    # optional: draw center if available (expects tuple (x, y))
    if acc.center:
        ax.plot(acc.center[0], acc.center[1], 'x', color="red", markersize=10)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()