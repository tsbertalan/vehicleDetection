from glob import glob
import matplotlib.image as mpimage
import tqdm

subKeys = {
    'vehicles': ['GTI_Far', 'GTI_Left', 'GTI_Right', 'GTI_MiddleClose'],
    'non-vehicles': ['GTI'],
}

def getData():


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
        mpimage.imread(path)
        for (i, veh, gti, path) in tqdm.tqdm_notebook(paths, unit='path')
    ]

    classes = [
        0 if veh == 'non-vehicles' else (subKeys['vehicles'].index(gti)+1)
        for (i, veh, gti, path)
        in paths
    ]

    return images, classes