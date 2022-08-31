import numpy as np
from skimage import feature


def get_grid(image, n_row=3, n_col=3):
    h = image.shape[0] // n_row
    w = image.shape[1] // n_col
    out = []
    for i in range(n_row):
        for j in range(n_col):
            out.append(image[i * h:(i + 1) * h, j * w:(j + 1) * w])
    return np.array(out)


class LBP:
    def __init__(self, numPoints, radius, num_bins=10, n_row=3, n_col=3):
        self.numPoints = numPoints
        self.radius = radius
        self.num_bins = num_bins
        self.n_row = n_row
        self.n_col = n_col

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image,
                                           self.numPoints,
                                           self.radius,
                                           method="default")
        grid_lbp = get_grid(lbp, self.n_row, self.n_col)
        final_hist = []
        for img in grid_lbp:
            (hist, _) = np.histogram(img.ravel(),
                                     bins=self.num_bins,
                                     range=(0, 2 ** self.numPoints))
            # normalize the histogram
            hist = hist.astype("float")
            hist /= (hist.sum() + eps)
            final_hist.append(hist)
        # return the histogram of Local Binary Patterns
        return np.array(final_hist).ravel(), lbp
