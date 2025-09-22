import numpy as np
import cv2
import matplotlib.pyplot as plt

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

    sx, sy = (pixel_pitch, pixel_pitch) if np.isscalar(pixel_pitch) else pixel_pitch

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

class CartesianAccumulator:
    def __init__(self, shape: tuple[int, int]):
        h, w = shape
        self.h, self.w = h, w
        self.counts = np.zeros((h, w), np.uint32)

    def update(self, img: np.ndarray, thresh: float | None = 128, invert: bool = False):
        if thresh is None:
            fg = (img != 0)
        else:
            fg = (img <= thresh) if invert else (img >= thresh)
        self.counts += fg.astype(np.uint8)

    def points(self,
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
        return pts, wts

    def remove_unwanted(self,
                        x_min: float, x_max: float,
                        y_min: float, y_max: float,
                        pixel_pitch: float | tuple[float, float] = 1.0,
                        centered: bool = True) -> None:
        """
        Zero-out counts for pixels whose Cartesian (x,y) lie within the inclusive
        rectangle [x_min,x_max] × [y_min,y_max].
        """
        sx, sy = (pixel_pitch, pixel_pitch) if np.isscalar(pixel_pitch) else pixel_pitch

        # Build pixel index grids
        rows = np.arange(self.h, dtype=np.float32)[:, None]   # shape (h,1)
        cols = np.arange(self.w, dtype=np.float32)[None, :]   # shape (1,w)

        if centered:
            cx, cy = (self.w - 1) / 2.0, (self.h - 1) / 2.0
            x = (cols - cx) * sx                    # +x right
            y = (cy - rows) * sy                    # +y up
        else:
            x = cols * sx
            y = -rows * sy

        inside = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
        self.counts[inside] = 0

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
