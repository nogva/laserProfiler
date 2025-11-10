from utils import *
from dotRocognition import *
from pixelToCartesian import *
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# config:
play_video = True

def is_image_path(p: str) -> bool:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return Path(p).suffix.lower() in exts

def process_frame(frame, kernel):
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(grey, 30, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

def main():

    acc = None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    if is_image_path(media_path):
        frame = cv2.imread(media_path, cv2.IMREAD_COLOR)
        assert frame is not None, "Cannot read image"
        height, width = frame.shape[:2]
        acc = CartesianAccumulator((height, width))

        mask = process_frame(frame, kernel)

        if play_video:
            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("frame", mask)
            # press any key (or q) to close
            key = cv2.waitKey(0)
            if key & 0xFF == ord('q'):
                pass

        pts = img_to_cartesian(
            img=mask,
            thresh=None,
            invert=False,
            pixel_pitch=1.0,
            centered=True
        )
        acc.update(mask, thresh=None)

    else:
        cap = cv2.VideoCapture(media_path, cv2.CAP_FFMPEG)
        assert cap.isOpened(), "Cannot open video"

        # fallback-safe fps
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 1e-3 else 30.0

        # read first frame to get dimensions and bootstrap accumulator
        ret, first = cap.read()
        assert ret and first is not None, "Cannot read first video frame"
        height, width = first.shape[:2]
        acc = CartesianAccumulator((height, width))

        # process first frame
        mask = process_frame(first, kernel)
        if play_video:
            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("frame", mask)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        pts = img_to_cartesian(
            img=mask,
            thresh=None,
            invert=False,
            pixel_pitch=1.0,
            centered=True
        )
        acc.update(mask, thresh=None)

        # main loop for remaining frames
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            mask = process_frame(frame, kernel)

            if play_video:
                cv2.imshow("frame", mask)
                if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                    break

            pts = img_to_cartesian(
                img=mask,
                thresh=None,
                invert=False,
                pixel_pitch=1.0,
                centered=True
            )
            acc.update(mask, thresh=None)

        cap.release()
        cv2.destroyAllWindows()

    # post-accumulation analysis
    acc.identify_bounds()
    acc.identify_center()
    #acc.remove_unwanted_around_center(radius_to_width_ratio=0.25)
    acc.point_metrics(should_print=True, scale=46/341.84)

    # plotting
    fig, ax = plt.subplots(figsize=(6, 6))
    pts = acc.pts
    if pts is None or pts.size == 0:
        raise ValueError("acc.pts is empty. Ensure acc.update(...) was called on at least one mask.")

    ax.scatter(pts[:, 0], pts[:, 1], s=10, marker='.', linewidths=0)

    if acc.bounds:
        xs, ys = zip(*(acc.bounds + [acc.bounds[0]]))
        ax.plot(xs, ys, linewidth=2)

    if acc.center:
        ax.plot(acc.center[0], acc.center[1], 'x', markersize=10)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
