import numpy as np
import cv2
import matplotlib.pyplot as plt

class CartesianAccumulator:
    def __init__(self, shape: tuple[int, int]):
        h, w = shape
        self.h, self.w = h, w
        self.counts = np.zeros((h, w), np.uint32)
        self.pts = None  # A 2D array of points (x,y) from self.points()
        self.bounds = None   # four Cartesian corner points
        self.center = None   # (x,y) center point

    def update(self, img: np.ndarray, thresh: float | None = 128, invert: bool = False):
        if thresh is None:
            fg = (img != 0)
        else:
            fg = (img <= thresh) if invert else (img >= thresh)
        self.counts += fg.astype(np.uint8)

    def update_points(self,
               pixel_pitch: float | tuple[float, float] = 1.0,
               centered: bool = True,
               min_count: int = 1):
        rows, cols = np.where(self.counts >= min_count)
        if rows.size == 0:
            return np.empty((0, 2), np.float32), np.empty((0,), np.uint32)

        sx, sy = (pixel_pitch, pixel_pitch) if np.isscalar(pixel_pitch) else pixel_pitch

        if centered:
            cx, cy = (self.w - 1) / 2.0, (self.h - 1) / 2.0
            x = (cols.astype(np.float32) - cx) * sx
            y = (cy - rows.astype(np.float32)) * sy
        else:
            x = cols.astype(np.float32) * sx
            y = -rows.astype(np.float32) * sy

        pts = np.column_stack((x, y))
        wts = self.counts[rows, cols]
        self.pts = pts
        return pts, wts

    def identify_bounds(self,
                        min_count: int = 1,
                        pixel_pitch: float | tuple[float, float] = 1.0,
                        centered: bool = True):
        rows, cols = np.where(self.counts >= min_count)
        if rows.size == 0:
            self.bounds = None
            return None

        rmin, rmax = int(rows.min()), int(rows.max())
        cmin, cmax = int(cols.min()), int(cols.max())

        if np.isscalar(pixel_pitch):
            sx = sy = float(pixel_pitch)
        else:
            sx, sy = float(pixel_pitch[0]), float(pixel_pitch[1])

        if centered:
            cx = (self.w - 1) / 2.0
            cy = (self.h - 1) / 2.0
            x_min = (cmin - cx) * sx
            x_max = (cmax - cx) * sx
            y_max = (cy - rmin) * sy
            y_min = (cy - rmax) * sy
        else:
            x_min = cmin * sx
            x_max = cmax * sx
            y_max = -rmin * sy
            y_min = -rmax * sy

        self.bounds = [
            (x_min, y_max),  # top-left
            (x_max, y_max),  # top-right
            (x_max, y_min),  # bottom-right
            (x_min, y_min)   # bottom-left
        ]
        return self.bounds

    def identify_center(self):

        if not self.bounds is None:
            x = (self.bounds[0][0]+self.bounds[1][0]) / 2
            y = (self.bounds[0][1]+self.bounds[2][1]) / 2
            self.center = (x, y)
        return
    
    def remove_unwanted_around_center(self, radius_to_width_ratio: float = 0.20):
        """
        Remove points whose Euclidean distance to the center is <= radius.
        Radius = ratio * bounding-box width (x-extent).
        """
        # Ensure we have bounds and center
        if self.bounds is None:
            self.identify_bounds()
        if self.bounds is None:
            return
        if self.center is None:
            self.identify_center()
        if self.center is None:
            return
        if self.pts is None:
            self.update_points()

        # radius from x-extent of bounds
        radius = radius_to_width_ratio * (self.bounds[1][0] - self.bounds[0][0])
        r2 = radius * radius

        # assume self.points shape (N, 2) as float or int
        cx, cy = self.center
        dx = self.pts[:, 0] - cx
        dy = self.pts[:, 1] - cy
        keep = (dx*dx + dy*dy) > r2  # keep points strictly outside radius

        # in-place shrink
        self.pts = self.pts[keep]


    def plot(self,
            pixel_pitch: float | tuple[float, float] = 1.0,
            centered: bool = True,
            min_count: int = 1,
            mode: str = "scatter",           # "scatter" or "density"
            max_points: int = 200_000,       # downsample for speed
            point_size: float = 1.0,
            figsize: tuple[int, int] = (6, 6),
            ax=None):
        """
        Visualize accumulated foreground.
        - mode="scatter": plots Cartesian points; sizes scale with counts.
        - mode="density": shows count image in Cartesian extent.
        """
        sx, sy = (pixel_pitch, pixel_pitch) if np.isscalar(pixel_pitch) else pixel_pitch

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if mode == "scatter":
            pts, wts = self.points(pixel_pitch=(sx, sy), centered=centered, min_count=min_count)
            n = pts.shape[0]
            if n == 0:
                ax.set_title("No points")
                ax.set_aspect('equal', 'box')
                return ax

            if n > max_points:
                idx = np.random.choice(n, size=max_points, replace=False)
                pts = pts[idx]
                wts = wts[idx]

            # size ~ log(count); avoids huge marker sizes
            sizes = point_size * (np.log1p(wts) / (np.log1p(wts).max() or 1.0)) * 10.0
            ax.scatter(pts[:, 0], pts[:, 1], s=sizes, linewidths=0)

            # Set symmetric/anchored limits for readability
            if centered:
                xlim = ((-(self.w - 1) / 2.0) * sx, ((self.w - 1) / 2.0) * sx)
                ylim = ((-(self.h - 1) / 2.0) * sy, ((self.h - 1) / 2.0) * sy)
            else:
                xlim = (0, (self.w - 1) * sx)
                ylim = (-(self.h - 1) * sy, 0)
            ax.set_xlim(xlim); ax.set_ylim(ylim)

        elif mode == "density":
            # Build Cartesian extent for imshow
            if centered:
                x_min, x_max = (-(self.w - 1) / 2.0) * sx, ((self.w - 1) / 2.0) * sx
                y_min, y_max = (-(self.h - 1) / 2.0) * sy, ((self.h - 1) / 2.0) * sy
            else:
                x_min, x_max = 0, (self.w - 1) * sx
                y_min, y_max = -(self.h - 1) * sy, 0

            # Show zeros as transparent by masking
            data = self.counts.astype(float)
            data[data == 0] = np.nan
            im = ax.imshow(
                data,
                origin="upper",               # row 0 at top; extent fixes orientation
                extent=[x_min, x_max, y_max, y_min]  # flip y so +y is up
            )
            # Optional: add a colorbar if desired in your environment
            # plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            raise ValueError('mode must be "scatter" or "density"')

        ax.set_aspect('equal', 'box')
        ax.grid(True, which='both', linestyle='--', color='lightgray', linewidth=0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return ax

    def point_metrics(self, decimals: int = 2, should_print: bool = False, scale: float = 0.1170):
        if self.pts is None or self.center is None:
            raise ValueError("Points or center not set. Run update_points(...) first and define self.center.")

        # distances from each point to the center
        dx = self.pts[:, 0] - self.center[0]
        dy = self.pts[:, 1] - self.center[1]
        dists = np.sqrt(dx**2 + dy**2) * scale

        metrics = {
            "mean_distance": round(float(np.mean(dists)), decimals),
            "min_distance":  round(float(np.min(dists)), decimals),
            "max_distance":  round(float(np.max(dists)), decimals),
            "std_distance":  round(float(np.std(dists)), decimals),
        }

        if should_print:
            print(f"Mean distance: {metrics['mean_distance']}")
            print(f"Min distance:  {metrics['min_distance']}")
            print(f"Max distance:  {metrics['max_distance']}")
            print(f"Std distance:  {metrics['std_distance']}")

        return metrics


def img_to_cartesian(
    img: np.ndarray,
    thresh: float | None = 128,
    invert: bool = False,
    pixel_pitch: float | tuple[float, float] = 1.0,
    centered: bool = True
) -> np.ndarray:
    """
    Convert a 2D image/mask to Cartesian points (N,2), +x right, +y up.

    Parameters
    ----------
    img : np.ndarray
        2D array. Convert BGR to gray before calling.
    thresh : float | None, default=128
        Include pixels >= thresh (<= if invert=True). If None, include nonzeros.
    invert : bool, default=False
        THRESH_BINARY_INV semantics.
    pixel_pitch : float | tuple[float, float], default=1.0
        Scalar or (sx, sy).
    centered : bool, default=True
        True → origin at symmetric center ((w-1)/2,(h-1)/2).
        False → origin at pixel (0,0).

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) with Cartesian coordinates.
    """
    assert img.ndim == 2, "img must be 2D"
    h, w = img.shape

    if isinstance(pixel_pitch, (tuple, list, np.ndarray)):
        sx, sy = pixel_pitch
    else:
        sx, sy = pixel_pitch, pixel_pitch

    if thresh is None:
        fg = (img != 0)
    else:
        fg = (img <= thresh) if invert else (img >= thresh)

    rows, cols = np.nonzero(fg)
    if rows.size == 0:
        return np.empty((0, 2), np.float32)

    if centered:
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        x = (cols.astype(np.float32) - cx) * sx
        y = (cy - rows.astype(np.float32)) * sy
    else:
        x = cols.astype(np.float32) * sx
        y = -rows.astype(np.float32) * sy

    return np.column_stack((x, y))