# Vehicle Detection

The goal of this project was to use classical computer vision methods (that is, not deep convolutional neural networks (CNN)) to detect and draw bounding boxes around vehicles which show up in footage from a hood-mounted camera. This can be reduced to a supervised classification task, in which image patches are classified as either "car" or "not-car". While, in a CNN approach, this scale/translation slewing (a covolution!) can be handled efficiently, with a network that directly outputs bounding boxes, or a detection heatmap, here, we make do with explicit windows.

Eliding some details, the classifier pipeline takes as input image patch, transforms from RGB to a more meaningful color space, computes higher-order features including color histograms and locally-binned histograms of oriented gradients (HOG), and then passes these to a radial basis function support vector classifier (RBF-SVC). Each detection window is added to a heat map over the entire image.

In order to encourage the surfacing of only persistent detections, this heat map is used in a simulated cooling process to produce a temperature map which evolves over multiple frames of video. The temperature map is thresholded to produce several disjoint regions, bounding boxes on which are taken to be the final detection bounding boxes.

## Training data

While multiple sources of data were available, for simplicity I used only the [GTI Vehicle Image Database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), from the Technical University of Madrid. This is a collection of 64x64 color images of cars, vans, trucks, etc. seen on the roads of Madrid, Brussels, and Turin.

![some GTI positive examples](VehicleDatabase.png)
*(Image taken from [www.gti.ssr.ump.es](http://www.gti.ssr.upm.es/data/Vehicle_database.html).)*

Both positive (shown above) and negative examples are included, and positive examples are further divided into far, left, middle-close, and right categories, which might be an interesting expansion on the basic car/no-car categorization done here. The basic dataset provided includes 3,900 negative examples and 2,826 positive examples.

### Train/test split for GTI data
Because GTI images are extracted from larger 360x256 video frames, the exhibit some seqentiality. Often, runs of up to a dozen images that are clearly of the same car occur in sequence in the data.
![sequential GTI images](sortedGTICarImages.png)
Here, numbers after the colon are the file names of the images, minus the prefix "image" and postfix ".png". Transitions in these supplied indices are indicated in red while it's apparent that some vehicle transitions are noted this way, by no means are all.

The presens of runs of the same car is significant for assessing training progress--when creating a train/test split of the data, a naively random split will include portions of most runs in both parts of the split. Since many images are very similar to others within their run, this will result in a testing dataset that is insufficiently different from the training set to provide a realistic idea of generalizability of results.

My first method for alleviating this problem, though unused, did provide some useful insight, and so I'll discuss it here. My goal was to detect these transitions, so that I could randomly assign whole runs to the train or test set. Using the color-histogram image featurization discussed below, I calculated the Euclidean feature distance between subsequent images. Simply thresholding these values was insufficient for separating same-car transitions from different-car, as visible by the evident difficulty of fitting a high-quality logistic regression to the feature distance data.
![can't do logistic regression here](logisticRegressionHard.png)

Instead, I considerd searching for local peaks in this signal, defined as distances which were a factor of 1.5 larger than the median of their radius-3 neighborhood to right and left.

![searching for car transitions](carTransitions.png)
Here, "true" transitions were found for the first 200 or so images by manual inspection, marked by black vertical lines. Red dashed lines correspond to jumps in the indexing implied by the image file names, while magenta lines are local peaks of the signal. This method likely would have performed well enough to produce an adequate train/test split, or even many such splits, suitable for automated k-fold cross validation when performing a grid search for parameter values (see below).

However, rather taking the time to write the data generator necessary to make full use of these splits for cross validation, for the purposes of once-off classifier validation, I instead simply used the first ~90% of the image from each class as training data, and the last ~10% as testing.


## Featurization
The first step in featurization of the image patches here was transformation to a more meaningful color space. While the default red-green-blue encoding of ingested images is sufficient for storage, the red, green, and blue channels for individual pixels are highly correlated, inhibiting parsimony in any classifier that uses these features. One way to see this is in the similarity of the mean red, green, and blue channels.
![mean RGB channels](meanCarImage.png)

Instead, I used the hue-lightness-saturation colorspace, in which different information is more readily available in the three channels. While extensive variation in the H and S channels makes the mean images uninterpretable, they are distinct from the mean L image.
![mean HLS channels](car.png)

Viewed on an individual image, however, the H and S channels  clearly pick out the taillights of a car, while the L channel shows the horizontal shape of its trunk and bumper.
![HLS on one image](sampleChannels.png)

In addition to the raw HLS pixel values, I used histogram of oriented gradient (HOG) features on each of the HLS channels. This method divides the image into cells a few pixels wide (usually 8x8 for my purposes), then allows each pixel to vote for one of a small number of orientations (here, usually 9). These votes are weighted by the strength of the gradient at that pixel, to produce a histogram of gradient directions within that cell. After a normalization involving several neighboring cells, these histograms are concatenated into a contribution towards the feature vector.

Visualized in the aggregate, the average HLS image shows a strong horizontal edge at the top of the L channel on positive-class images. At the moment, my classifiers all seem to rely heavily on this feature, perhaps to the exclusion of light-colored cars (whose upper edge gradient might be reversed in direction).
![mean HLS HOG](car%20HOG.png)


### Data augmentation.
Since the GTI data seems to be generally lower-resolution and darker European images, compared the gloriously sunny high-resolution southern California video of the test videos, I was initially quite concerned that the simple distribution of pixel values would be quite different between the train/test and inference data. Below I show these distributions in HSV space, which is similar to the distribution in HSL.

![training pixel distribution](distTrain.png)
*Distribution of training pixels.*

![training pixel distribution](distTest.png)
*Distribution of inference pixels.*

I was particularly concerned about the large gap in the 40-80 range of the H channel. However, upon examining the color that these values corespond to in the 180-double-degree encoding used by OpenCV, I was able to hand wave this as a preponderance of negative-class foilage images in the training data.

However, I was still concerned that the inference video frames seemed subjectively lighter than the training images, so I implemented a data agumentation scheme, whereby I converted some subset of the training images to HSV space, increased the V channel a random amount such than no pixel's V channel was allowed to go over 255, then converted back to RGB.

Additionally, I did some simple left-right flip augmentation.

It's not clear to me that this augmentation necessarily improved the classification results, but it certainly did make training slower.


### HOG reuse
While HOG features were generated from scratch each time for training images, in order to save time and memory during inference, I developed a method to generate HOG features across the whole image only once, reusing this for all translated and scaled image windows. While this slightly tightened the coupling of the code, in combination of vectorization of the inference operation, this produced a 10x speedup at test time.

When called with `feature_vector=False, orient=9, pixels_per_cell=8`, and `cells_per_block=2`, the featurizer of `skimage.feature.hog` returns a `feature_array` of shape `(nx, ny, 2, 2, 9)`, where the `nx` and `ny` components represent the number of times the `cells_per_block` by `cells_per_block` normalization block can be translated across the image in steps of once cell. To pregenerate my HOG features, I first took horizontal-slice regions-of-interest from my input frames, then expressed my goal image-patch radius for each slice (e.g. 128), as well as fractional overlap between patches (e.g. 0.5). I scaled the slices up or down such that the desired patch sizes were mapped to the 64x64 size of my training patches, and applied `skimage.feature.hog`. I then found indexes into the first two dimensions of `feature_array` such that 64x64 patches of the desired fractional overlap were approximately achieved, considering the stride pattern of the normalization blocks. While unrestricted use of the parameters for `skimage.feature.hog` will prevent the number and size (when mapped back to video pixel space) of the windows generated this way from being exactly as requested, they will still map to patches of the same number of HOG features, and, for a given video frame size, will always produce the same number of trial windows. As a final step, I computed the resulting indexes into the original image, and also returned windowed views of that image for computing raw-pixel (or color histogram) features.

As a special case, a goal of a 64x64 (unscaled) window, applied to a full-height "slice" of a 64x64 input image produces only a single window and the HOG features thereof. This allowed me to use the same code at train and test time.

All featurization code is contained in a single `FeatureExtractor` class, whose `__call__` method dispatches between the 64x64 single-window special-case and the large-frame, many-window inference case; returning feature vector or array of feature vectors, accompanied by window boundaries, as appropriate.



## Classification
Though perhaps it was a premature decision, I quickly settled on a RBF-SVC as my classification method, as this seemed to provide equal-quality results to a linear-kernel SVC or a decision tree on my initial tests, with better speed than the second and surprisingly similar speed to the first.

I used `sklearn`'s built-in cross-validated grid search to find values for the $C$ and $\gamma$ parameters of the RBF-SVC. A more careful approach would include the choice among the three classifiers in this grid search (which might need to become a random hyperparameter search, or a more sophisticated Bayesian hyperparameter search, as the dimension of the hyperparameter space grows). Additionally, due to the sequential nature of the data discussed above, the validation errors used for this search should be taken with a massive grain of salt. Really, the method described in the "train/test split for GTI data" section above should be used for generating the train-test splits needed for the k-fold cross-validation strategy used by `sklearn`.

Regardless, this search *did* enable met to find a workable region of the $C\times\gamma$ space in which the classifier wouldn't completely collapse to predicting all one class or the other.

![RBF-SVC hyperparameter search](hyperparameterSearch.png)
