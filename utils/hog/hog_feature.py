from skimage import exposure
from skimage import feature


class HOG:
    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3)):
        # store the number of points and radius
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def describe(self, image):
        (H, hogImage) = feature.hog(image, self.orientations, self.pixels_per_cell,
                                    self.cells_per_block, transform_sqrt=True, block_norm="L2",
                                    visualize=True)
        hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
        hogImage = hogImage.astype("uint8")
        return H, hogImage
