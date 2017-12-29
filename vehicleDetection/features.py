import numpy as np
from skimage.feature import hog
import cv2

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


def get_hog_features(
    img, orient, pix_per_cell, cell_per_block, 
    vis=False, feature_vec=True
    ):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features, None


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


def multichannelHog(
    feature_image, 
    hog_channel,
    orient,
    pix_per_cell,
    cell_per_block,
    vis=False,
    ):
    if hog_channel == 'ALL':
        hog_features = []
        hogVis = []
        for channel in range(feature_image.shape[2]):
            hog_featuresChannel, hogVisChannel = get_hog_features(
                feature_image[:, :, channel], 
                orient, pix_per_cell, cell_per_block, 
                vis=vis, feature_vec=True
            )
            hog_features.append(hog_featuresChannel)
            hogVis.append(hogVisChannel)
        hog_features = np.ravel(hog_features)
        hogVis = np.dstack(hogVis)
    else:
        hog_features, hogVis = get_hog_features(
            feature_image[:, :, hog_channel], 
            orient, pix_per_cell, cell_per_block,
            vis=vis, feature_vec=True
        )
    return hog_features, hogVis


class FeatureExtractor:

    def __init__(self, 
        color_space='HLS', spatial_size=(32, 32),
        hist_bins=32, orient=9, 
        pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
        spatial_feat=True, hist_feat=True, hog_feat=True
        ):
        self.color_space = color_space
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel
        self.spatial_feat = spatial_feat
        self.hist_feat = hist_feat
        self.hog_feat = hog_feat

    def __call__(self, imgOrImgs):
        """Extract features from a list of images.
        Have this function call bin_spatial() and color_hist()
        """
        if isinstance(imgOrImgs, list):
            return [
                self(img) for img in imgOrImgs
            ]
        else:
            return self._extract_features(imgOrImgs)[0]

    def hogVis(self, image):
        return self._extract_features(image, True)[1]

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

    def _extract_features(self, image, vis=False):

        # Disambiguate uint8 [0, 255] or float [0, 1] images.
        if image.dtype == np.uint8 or (image > 1).any():
            image = (np.copy(image) / 255.).astype('float32')

        # Get color channels.
        feature_image = self.getChannels(image)

        # Extract all the requested features.
        features = []

        if self.spatial_feat:
            spatial_features = bin_spatial(feature_image, size=self.spatial_size)
            features.append(spatial_features)

        if self.hist_feat:
            hist_features = color_hist(feature_image, nbins=self.hist_bins)
            features.append(hist_features)

        if self.hog_feat:
            hog_features, hogVis = self._hog(
                feature_image,
                vis=vis,
            )
            features.append(hog_features)

        return np.concatenate(features), hogVis

    def _hog(self, feature_image, vis=False):
        return multichannelHog(
            feature_image,
            self.hog_channel,
            self.orient,
            self.pix_per_cell,
            self.cell_per_block,
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
        spatial_features = bin_spatial(feature_image, size=self.spatial_size)
        multiLinePlot(spatial_features, 'image')

        # Show the color histogram features.
        hist_features = color_hist(feature_image, nbins=self.hist_bins)
        multiLinePlot(hist_features, 'color hist')

        # Show the HOG features.
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
                ax.set_xticks([])
                if ax not in keepTicks:
                    ax.set_yticks([])

        return fig, axes
