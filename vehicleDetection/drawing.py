import numpy as np
import cv2


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def labeledBboxes(labels):
    bboxes = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        bboxes.append(bbox)

    return bboxes


def drawBboxes(img, bboxes, color=(0, 0, 255)):
    img = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(img, bbox[0], bbox[1], color, 6)
    return img


def draw_labeled_bboxes(img, labels, **kw):
    bboxes = labeledBboxes(labels)
    return drawBboxes(img, bboxes, **kw)


def writeText(img, text, pixelsPerColumn=13, roffset=30, rpix=80, cpix=10, copy=True, **kwargs):
    if copy:
        img = np.copy(img)

    nrows, ncols = img.shape[:2]
    for k, v in dict(
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=3,
        thickness=2,
        color=(255, 255, 0),
        lineType=cv2.LINE_AA,
        ).items(): 
        kwargs.setdefault(k, v)

    pixScale = (kwargs['fontScale'] / .75)
    wrapcols = int(ncols / pixelsPerColumn / pixScale)

    # Wrap lines.
    import textwrap
    textlines = textwrap.wrap(text, width=wrapcols)

    for line in textlines: 
        cv2.putText(
                img, 
                line,
                (cpix, rpix),
                **kwargs
            )
        rpix += int(roffset * pixScale)

    return img
