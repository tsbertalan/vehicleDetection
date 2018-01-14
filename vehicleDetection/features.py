import numpy as np

from skimage.feature import hog
import cv2


import logging
# create logger with 'spam_application'
logger = logging.getLogger('featurize')
logger.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(ch)

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


d = 64


def ar(f):
    fr = round(f)
    assert f == fr
    return int(fr)


def breakIntoWindows(
    feature_channel,
    scales = None,
    orient = 9,
    pixels_per_cell = 8,
    cells_per_block = 2,
    visualize=False,
    ):
    if scales is None:
        scales = [
        #  scale,    (lo,  hi),  overlap
           #(256/64, (720, 400), .5),
            (d/128, (690, 400), .5),
            (d/96,  (600, 400), .4),
            (d/64,  (600, 400), .3),
            (d/48,  (550, 400), .25),
        ]
    
    fl = np.math.floor
    
    # Accumulate output.
    blockWindows = []
    windowLocations = []
    sliceWindows = []
    hogVisualizations = []
    
    for islice in range(len(scales)):
        
        # Extract the image slice and goal window geometry.
        scale, (lo, hi), overlapFraction = scales[islice]

        # Resize the image slice so that the windows will be d-by-d.
        assert lo > hi
        unscaledSlice = feature_channel[hi:lo, :]
        sliceShape = tuple([int(s*scale) for s in unscaledSlice.shape[:2]])
        imgSlice = cv2.resize(unscaledSlice, sliceShape[::-1])
        
        # What is the geometry of a window and stride in the unscaled image?
        overlapPix = min(fl(d * overlapFraction), d - 1)
        stridePix = d - overlapPix
            
        # Extract HOG features for the scaled slice.
        res = hog(
            imgSlice[:, :],
            orientations=orient,
            pixels_per_cell=(pixels_per_cell, pixels_per_cell), 
            cells_per_block=(cells_per_block, cells_per_block),
            visualise=visualize, feature_vector=False,
            block_norm='L2-Hys',
        )
        feature_array, hog_image = res if visualize else (res, None)

        # How do we convert from cell-&-block indexing to unscaled-pixel indexing?
        cellsPerWindow = ar(d / pixels_per_cell)
        
        # How does sklearn do the HOG block striding?
        overlappingBlocksPerWindow = cellsPerWindow - 1
        cellsPerBlockStride = 1
        
        # How many blocks should *we* step by so our windows overlap
        # by approximately the desired fraction?
        strideBlocks = round(stridePix / pixels_per_cell / cellsPerBlockStride)
        
        # Stride.
        dr = dc = overlappingBlocksPerWindow
        rs = cs = strideBlocks
        rpix = 0
        rl = 0
        nr, nc = feature_array.shape[:2]
        while True:
            cl = 0
            cpix = 0
            while True:
                
                # Extract the HOG window.
                blockWindow = feature_array[rl:(rl+dr), cl:(cl+dc), ...]
                blockWindows.append(blockWindow)
                
                # Extract the image window.
                wl = np.floor(np.array((
                    (cpix / scale, hi + rpix / scale), ((cpix + d) / scale, hi + (rpix + d) / scale)
                ))).astype(int)
                windowLocations.append(tuple([tuple(p) for p in wl]))
                sliceWindows.append(
                    imgSlice[rpix:rpix+d, cpix:cpix+d]
                )
                
                # Extract the HOG visualization window.
                if visualize:
                    hogVisualizations.append(
                        hog_image[rpix:rpix+d, cpix:cpix+d]
                    )
                    
                # Increment the indices.
                if cl + cs >= nc or cl + cs + dc > nc:
                    break
                else:
                    cl += cs
                    cpix += stridePix
                    
            if rl + rs >= nr or rl + rs + dr > nr:
                break
            else:
                rl += rs
                rpix += stridePix
            
    return blockWindows, sliceWindows, windowLocations, hogVisualizations


class FeatureExtractor:

    def __init__(self, 
        color_space='HLS', spatial_size=(32, 32),
        hist_bins=32, orient=9, 
        pixels_per_cell=8, cells_per_block=2, hog_channel='ALL',
        spatial_feat=True, hist_feat=False, hog_feat=True
        ):
        self.color_space = color_space
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.orient = orient
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.hog_channel = hog_channel
        self.spatial_feat = spatial_feat
        self.hist_feat = hist_feat
        self.hog_feat = hog_feat

    @property
    def colorSpaceNames(self):
        if len(self.color_space) == 3:
            c = list(self.color_space)
        else:
            assert self.color_space == 'YCrCb'
            c = ['Y', 'Cr', 'Cb']
        if self.hog_channel != 'ALL':
            c = [c[self.hog_channel]]
        return c

    def getChannels(self, image):
        """Apply color conversion if other than 'RGB'."""
        color_space = self.color_space
        if self.color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)
        return feature_image

    def __call__(self, image, window=True, giveWindows=None):

        if window:
            assert image.shape[:2] == (64, 64)

        if image.dtype != 'uint8' or (image <= 1).all():
            logger.warning('Image was not converted to UINT8!')
            image = (image * 255).astype('uint8')

        # Disambiguate uint8 [0, 255] or float [0, 1] images.
        if image.dtype == np.uint8 or (image > 1).any():
            image = (np.copy(image) / 255.).astype('float32')

        # Get color channels.
        feature_image = self.getChannels(image)

        # Get windows and hog features.
        allHogWindows, allColorWindows, allWindowLocations = self.windowFeatures(
            feature_image, scales=None if not window else [(1, (64, 0), 0)]
        )

        features = [
            self._extract_features(color, hog)
            for (hog, color) in zip(allHogWindows, allColorWindows)
        ]
        giveWindows = not window
        if giveWindows:
            return np.stack(features), allWindowLocations
        else:
            return features[0]

    def windowFeatures(self, feature_image, scales=None):
        windows = [
            # blockWindows, sliceWindows, windowLocations, hogVisualizations:
            breakIntoWindows(
                feature_image[:, :, i],
                orient=self.orient,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                scales=scales,
            )
            for i in range(3)
        ]
        tricolorHogWindows, colorWindows = [
            np.stack([windows[channel][i] for channel in range(3)], axis=-1)
            for i in range(2)
        ]
        windowLocations = windows[0][2]
        return tricolorHogWindows, colorWindows, windowLocations
        
    def _extract_features(self, feature_image, hogChannels):
        
        # Extract all the requested features.
        features = []

        if self.spatial_feat:
            spatial_features = bin_spatial(feature_image, size=self.spatial_size)
            features.append(spatial_features)

        if self.hist_feat:
            hist_features = color_hist(feature_image, nbins=self.hist_bins)
            features.append(hist_features)

        if self.hog_feat:
            features.append(hogChannels)

        return np.concatenate([f.ravel() for f in features])

    def _hog(self, feature_image, vis=False):
        return multichannelHog(
            feature_image,
            self.hog_channel,
            self.orient,
            self.pixels_per_cell,
            self.cells_per_block,
            vis=vis,
        )

    def visualize(self, image):
        import matplotlib.pyplot as plt

        if self.hog_channel == 'ALL':
            channelIndices = range(3)
        else:
            channelIndices = [self.hog_channel]

        # Generate a figure and determine how many subplot rows it will have.
        nrows = 1
        if self.spatial_feat: nrows += 1
        if self.hist_feat:    nrows += 1
        if self.hog_feat:     nrows += 1
        fig = plt.figure(figsize=(16, 3*nrows))
        keepTicks = set()

        rowIndex = [0]
        axes = []
        def generateRow(ncols):
            out = [
                fig.add_subplot(
                    nrows, 
                    ncols, 
                    rowIndex[0] * ncols + k + 1
                )
                for k in range(ncols)
            ]
            for ax in out:
                ax.set_title('row %d' % rowIndex[0])
            rowIndex[0] += 1
            axes.append(out)
            return out

        # Show the input.
        row = generateRow(len(channelIndices) + 1)
        row[0].imshow(image)
        row[0].set_title('color image')
        feature_image = self.getChannels(image)

        # Plotting utilities.
        breakChannels = lambda colors: [colors[:, :, i] for i in channelIndices]
        def multiLinePlot(signal, label):
            row = generateRow(len(channelIndices))
            C = self.colorSpaceNames
            for i in range(len(channelIndices)):
                ax = row[i]
                d = len(signal) / len(channelIndices)
                assert int(d) == d
                d = int(d)
                y = signal[d*i:d*(i+1)]
                ax.plot(y, linewidth=1, color='rgb'[i])
                ax.set_title('%s: %s' % (label, C[i]))
            [keepTicks.add(ax) for ax in row]
            return row
        
        # Show the color channels.
        channels = breakChannels(feature_image)
        for channel, ax, cname in zip(channels, row[1:], self.colorSpaceNames):
            if cname == 'H':
                cmap = 'hsv'
            else:
                cmap = 'viridis'
            ax.imshow(channel, cmap=cmap)
            ax.set_title(cname)

        # Show the spatial features.
        if self.spatial_feat:
            spatial_features = bin_spatial(feature_image, size=self.spatial_size)
            multiLinePlot(spatial_features, 'image')

        # Show the color histogram features.
        if self.hist_feat:
            hist_features = color_hist(feature_image, nbins=self.hist_bins)
            multiLinePlot(hist_features, 'color hist')

        # Show the HOG features.
        if self.hog_feat:
            row = generateRow(len(channelIndices) + 1)
            hogVis = self._hog(feature_image, vis=True)[1]
            row[0].imshow(hogVis)
            row[0].set_title('HOG')
            channels = breakChannels(hogVis)
            for channel, ax, cname in zip(channels, row[1:], self.colorSpaceNames):
                ax.imshow(channel)
                ax.set_title('HOG: %s' % cname)

        # Clean up the axes.
        for row in axes:
            for ax in row:
                if ax not in keepTicks:
                    ax.set_xticks([])
                    ax.set_yticks([])

        return fig, axes
