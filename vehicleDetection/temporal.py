import vehicleDetection
from vehicleDetection.videoUtils import *

import os


def intApproxSqrt(n):
    i = 1
    while i <= n:
        pass



def tileImages(*frames):
    pass
    


class HeatVideo:

    def __init__(
        self,
        fpath='../../../data/vehicleDetection/test_video.mp4',
        thr=1,
        **loadKwargs
    ):
        loadKwargs.setdefault('pbar', False)
        self.fpath = fpath
        self.baseLabel = os.path.basename(fpath).replace('.mp4', '')
        self.inputFrames = loadVideo(fpath=self.fpath, **loadKwargs)
        self.thr = thr
        self.coolingKwargs = {}

    def go(self, detector):

        if hasattr(detector.clf, 'n_support_'):
            print('Number of support vectors for each class:', detector.clf.n_support_)

        def generate(frame):
            out = detector.detect(frame, retDict=True, threshold=2)
            def check(labels=None, heatmap=None, bboxes=None):
                assert (heatmap >= 0).all()
                assert heatmap.dtype == float
                heatmap /= heatmap.max()
                heatmap *= 255
                heatmap = heatmap.astype('uint8')
                return bboxes, heatmap, labels
            return check(**out)

        self.heatSources = []
        self.bboxes = []
        self.labels = []
        for frame in tqdm.tqdm_notebook(self.inputFrames, desc='processing', unit='frame'):
            bboxes, heatmap, labels = generate(frame)
            self.heatSources.append(heatmap)
            self.bboxes.append(bboxes)
            self.labels.append(labels)

    def video(self, label=None, outVidPath=None):

        # Assemble output path.
        if outVidPath is None:
            outVidPath = 'doc/%s-detected.mp4' % (
                self.baseLabel,
            )
        if label is not None:
            ext = outVidPath[-4:]
            outVidPath.replace(ext, label + ext)

        framesToConcatenate = [
            # self.inputFrames, 
            self.cooling(**self.coolingKwargs),
            [
                vehicleDetection.drawing.draw_labeled_bboxes(
                    boxFrame, labels
                )
                for (boxFrame, labels) in zip(self.inputFrames, self.labels)
            ]
        ]

        return saveVideo(
            (
                np.hstack([heat, rawBoxes])
                for (
                    # frame, 
                    heat, 
                    rawBoxes, 
                    # persistentBoxes
                    )
                in zip(*framesToConcatenate)
            ),
            outVidPath
        )

    def cooling(self, heatTransferCoefficient=9, heatCapacity=10, dt=1):
        """Model exponential heat decay with source injections.

        Model heat decay at each pixel using
            dudt = (heatTransferCoefficient * A * (0 - u(t)) + h(t)) / heatCapacity
        where A is assumed to be 1 m**2

        Parameters
        ----------
        heatTransferCoefficient : float
            Rate at which heat leaks from each pixel; units are W/m**2/K
        heatCapacity : float
            How much thermal inertia the pixels have relative to
            heating from detections; units are J/K
        dt : float
            Duration of each step in the simulation; units are s, but not actual video s.
        """
        def decay(u0, power):
            """Euler time stepper/flow map"""
            dudt = (power - heatTransferCoefficient * u0) / heatCapacity
            return u0 + dudt * dt

        # Integrate the ODEs.
        u = [np.zeros_like(self.heatSources[0])]
        for heat in self.heatSources:
            u.append(decay(u[-1], heat))

        u = np.stack(u)
        u /= u.max()
        u *= 255
        u = u.astype('uint8')
        return u


class DetectedVideo:

    def __init__(
        self,
        fpath='../../../data/vehicleDetection/test_video.mp4',
        thr=1,
        **loadKwargs
    ):
        loadKwargs.setdefault('pbar', False)
        self.fpath = fpath
        self.inputFrames = loadVideo(fpath=self.fpath, **loadKwargs)
        self.thr = thr

    def go(self, det, label):

        self.detectedFrames = []
        fig, ax = plt.subplots()
        def draw(frame):
            det.drawHeat(frame, threshold=self.thr, ax=ax)
            out = fig2img(fig)
            self.detectedFrames.append(out)
            ax.cla()
            return out

        return saveVideo(
            (
                draw(frame)
                for frame in tqdm.tqdm_notebook(self.inputFrames, desc='bounding', unit='frame')
            ),
            'doc/%s-detected%s.mp4' % (
                os.path.basename(self.fpath).replace('.mp4', ''),
                label,
            )
        )

