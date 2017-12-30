import cv2
import numpy as np
from glob import glob
import matplotlib.image as mpimage
import tqdm

subKeys = {
    'vehicles': ['GTI_Far', 'GTI_Left', 'GTI_Right', 'GTI_MiddleClose'],
    'non-vehicles': ['GTI'],
}


def readImage(filePath):

    data = mpimage.imread(filePath)

    if filePath.endswith('.png'):
        assert data.max() <= 1.0
        assert data.min() >= 0.0
        data = (data*255).astype('uint8')

    return data


def randomLighten(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    factor = np.random.uniform(
        low=1, 
        high=255 / hsv[:, :, 2].max()
    )
    hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(float) * factor, 0, 255).astype('uint8')
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def getData(numLighter=2048):

    paths = []
    for mainKey in subKeys.keys():
        for subKey in subKeys[mainKey]:
            basePath = './data/%s/%s/%s/*.png' % (mainKey, mainKey, subKey)
            paths.extend([
                (
                    int(path.split('/')[-1].replace('.png', '').replace('image', '')),
                    mainKey,
                    subKey,
                    path,
                )
                for path in glob(basePath)
            ])

    ids = [x[0] for x in paths]

    images = [
        readImage(path)
        for (i, veh, gti, path) in tqdm.tqdm_notebook(paths, unit='path')
    ]

    classes = [
        0 if veh == 'non-vehicles' else (subKeys['vehicles'].index(gti)+1)
        for (i, veh, gti, path)
        in paths
    ]

    if numLighter > 0:
        
        indices = np.random.choice(np.arange(len(images)), size=numLighter, replace=False)
        lighterImages = [randomLighten(images[i]) for i in indices]
        ligherClasses = [classes[i] for i in indices]
        images += lighterImages
        classes += ligherClasses

    return images, classes