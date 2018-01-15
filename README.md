# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, the goal was to write a software pipeline to detect vehicles in a video.  The main output was a detailed writeup of the project, visible in [doc/report.md](doc/report.md) and [doc/report.html](doc/report.html). The writeup is written in markdown, but a makefile is provided in `doc/` to generate an HTML file.

![sample output](doc/samplePredictions.png)

From the report: 

The goal of this project was to use classical computer vision methods (that is, not deep convolutional neural networks (CNN)) to detect and draw bounding boxes around vehicles which show up in footage from a hood-mounted camera. This can be reduced to a supervised classification task, in which image patches are classified as either "car" or "not-car". While, in a CNN approach, this scale/translation slewing (a covolution!) can be handled efficiently, with a network that directly outputs bounding boxes, or a detection heatmap, here, we make do with explicit windows.

Eliding some details, the classifier pipeline takes as input image patch, transforms from RGB to a more meaningful color space, computes higher-order features including color histograms and locally-binned histograms of oriented gradients (HOG), and then passes these to a radial basis function support vector classifier (RBF-SVC). Each detection window is added to a heat map over the entire image.

In order to encourage the surfacing of only persistent detections, this heat map is used in a simulated cooling process to produce a temperature map which evolves over multiple frames of video. The temperature map is thresholded to produce several disjoint regions, bounding boxes on which are taken to be the final detection bounding boxes.
