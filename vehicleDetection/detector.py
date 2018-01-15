import numpy as np
import cv2
import matplotlib.pyplot as plt
import tqdm

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import vehicleDetection

def cached(method):
    def cachedMethod(self, *args, **kwargs):
        if not hasattr(self, '_cache'):
            self._cache = {}
        out = self._cache.get(
            method.__name__,
            None
        )
        if out is None:
            out = method(self, *args, **kwargs)
        self._cache[method.__name__] = out
        return out
    return cachedMethod

class Detector:

    def __init__(self, 
        slide_window_kwargs={}, 
        CLF=SVC, 
        clfParameters=dict(
            kernel='rbf',
            # High C is prone to overfitting; low, under.
            # Low C makes for a smoother decision surface.
            #
            # Increasing C roughly decreases the number of support vectors,
            # and speed of inference is rougly linear with this.
            # See https://stats.stackexchange.com/questions/270187
            #
            # C=15.199110829529332,
            # C=[10**(-.91)],
            C=1,
            # For radial basis functions, gamma is the inverse square 
            # of the kernel bandwidth--
            # the characteristic distance in feature space used for the
            # the implict diffusion transformation used before classification.
            #
            # High gamma means influence of neighbors is more local; low, global.
            # Low-gamma behaves more like a a linear SVC; high more complex.
            # gamma=6.5793322465756827e-05,
            gamma=7e-5,
        ),
        searchParams=dict(n_jobs=7, n_iter=512),
        #featurizeKwargs=dict(color_space='HSV'),
        featurizeKwargs={},
        scales = [
        #  scale, (lo,  hi), overlap
           #(256, (720, 400), .5),
            (128, (690, 400), .5),
            (96,  (600, 400), .5),
            (64,  (600, 400), .5),
            (48,  (550, 400), .5),
        ]
        ):
        self.scaler = StandardScaler()
        if clfParameters is None:
            self.clf = CLF()
        else:
            # If scalar parameters given, don't do a CV approach.
            anyScalars = False
            for k, v in clfParameters.items():
                try:
                    len(v)
                except TypeError:
                    anyScalars = True
                    break
            if anyScalars:
                print('Got scalar parameter (%s). Assuming no CV requested.' % k)
                # for k in clfParameters.keys():
                #     clfParameters[k] = [clfParameters[k]]
                self.clf = CLF(**clfParameters)

            else:
                # Count the number of combinations to decide
                # whether we should do a RandomizedSearchCV.
                npar = 1
                for v in clfParameters.values():
                    npar *= len(v)
                if npar > searchParams.get('n_iter', 32):
                    print('Using RandomizedSearchCV.')
                    Search = RandomizedSearchCV
                else:
                    print('Using GridSearchCV.')
                    searchParams.pop('n_iter', None)
                    Search = GridSearchCV
                self.clf = Search(CLF(), clfParameters, **searchParams)

        self.scales = scales
        self.slide_window_kwargs = {}
        self.slide_window_kwargs.update(slide_window_kwargs)
        self.featurize = vehicleDetection.features.FeatureExtractor(
            **featurizeKwargs
        )

    def save(self, fpath='data/detector.npz', addLabel=True):
        if addLabel and hasattr(self, 'ntrain'):
            fpath = '%s-%d_train.npz' % (fpath[:-3], self.ntrain)
            
        print('Saving to', fpath, '...', end=' ')
        np.savez(fpath, scaler=self.scaler, clf=self.clf)
        print('done.')

    def load(self, fpath='data/detector.npz'):
        data = np.load(fpath)
        self.scaler = data['scaler'].reshape((1,))[0]
        self.clf = data['clf'].reshape((1,))[0]

    def fit(self, imageWindows, classes, splitFrac=.9, **skw):
        features = np.vstack([
            self.featurize(image).reshape((1, -1))
            for image in imageWindows
        ])
        scaled = self.scaler.fit_transform(features)
        boolLabels = np.array(classes).astype(bool)#.reshape((-1, 1))

        # Generate train/test split.
        # Where do we change classes?
        divisions = (np.argwhere(np.diff(boolLabels)) + 1).ravel()

        # If we switch classes a lot, just do the simple split.
        oneClassCheck = True
        if len(divisions) > 10:
            oneClassCheck = False
            divisions = [int(len(boolLabels)*splitFrac)]

        # Regardless, add virtual indices at start and end.
        classBlocks = [0]
        classBlocks.extend(divisions)
        classBlocks.append(None)

        # Find the start and end indices of each same-class block,
        # and split the block into train and test portions.
        train = dict(feat=[], clas=[])
        test = dict(feat=[], clas=[])
        for i in range(len(classBlocks)-1):
            blockStart = classBlocks[i]
            blockEnd = classBlocks[i+1] 
               
            # Extract the same-class block.
            feat = scaled[blockStart:blockEnd]
            clas  = boolLabels[blockStart:blockEnd]
            if oneClassCheck:
                assert len(set(clas)) == 1, (set(clas), blockStart, blockEnd)
            
            # Put some of the block in train and some in test.
            split = int(len(feat) * splitFrac)
            train['feat'].extend(feat[:split])
            train['clas'].extend(clas[:split])
            test ['feat'].extend(feat[split:])
            test ['clas'].extend(clas[split:])

        X_train = train['feat']
        X_test  = test ['feat']
        y_train = train['clas']
        y_test  = test ['clas']

        # Simple split doesn't work if data is already sorted by class.
        # split = int(len(boolLabels) * splitFrac)
        # X_train = scaled[:split]
        # y_train = boolLabels[:split]
        # X_test = scaled[split:]
        # y_test = scaled[split:]

        # Random split doesn't work if there are many very similar examples
        # (e.g., images of the same car over time)
        # rand_state = 4
        # X_train, X_test, y_train, y_test = train_test_split(
        #     scaled, boolLabels,
        #     test_size=0.1, 
        #     random_state=rand_state
        # )

        self.ntrain = len(y_train)
        self.clf.fit(X_train, y_train)
        # Check the score of the SVC
        print(
            '%d-image train accuracy of clf = ' % len(y_train), 
            self.clf.score(X_train, y_train),
        )
        print(
            '%d-image test accuracy of clf = ' % len(y_test), 
            self.clf.score(X_test, y_test),
        )

        import sklearn.model_selection._search
        if isinstance(self.clf, sklearn.model_selection._search.BaseSearchCV):
            print('Best parameters:', self.clf.best_params_)

        self.save(**skw)

    def rawDetect(self, image):
        # Get features and corresponding image windows.
        windowFeatures, windowLocations = self.featurize(image, window=False)

        # Scale extracted features to be fed to classifier.
        test_features = self.scaler.transform(windowFeatures)
        
        # Predict using trained classifier.
        predictions = self.clf.predict(test_features)

        # Return positive detection windows.
        return [
            window 
            for (pred, window) in zip(predictions, windowLocations)
            if pred
        ]

    def drawDetect(self, image, ax=None, cleanax=True):
        bboxes = self.rawDetect(image)
        if ax is None: fig, ax = plt.subplots(figsize=(16,9))
        ax.imshow(
            vehicleDetection.drawing.draw_boxes(
                image, bboxes,
                # thick=1 will exclude some horizontals sometimes.
                thick=2,
            )
        )
        if cleanax:
            ax.set_xticks([])
            ax.set_yticks([]);
        return ax
