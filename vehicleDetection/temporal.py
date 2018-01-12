import vehicleDetection
from vehicleDetection.videoUtils import *

import os

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

    def go(self, detector):

        if hasattr(detector.clf, 'n_support'):
            print('Number of support vectors for each class:', detector.clf.n_support_)

        self.heatSources = []

        def draw(frame):
            heat = detector.heat(frame.shape, detector.rawDetect(frame))
            assert (heat >= 0).all()
            assert heat.dtype == float
            heat /= heat.max()
            heat *= 255
            heat = heat.astype('uint8')
            return heat

        for frame in tqdm.tqdm_notebook(self.inputFrames, desc='heating', unit='frame')
            self.heatSources.append(draw(frame))

    def heatSourceVideo(self, label=None, outVidPath=None):
        if outVidPath is None:
            outVidPath = 'doc/%s-heat.mp4' % (
                self.baseLabel,
            )
        if label is not None:
            ext = outVidPath[-4:]
            outVidPath.replace(ext, label + ext)

        return saveVideo(
            self.heatSources,
            outVidPath
        )

    def heatDecay(self, outVidPath=None, decayRate=.5, heatRate=1, dt=1):
        """Model exponential decay with source injections."""
        def decay(u0, heat):
            """Euler time stepper/flow map"""
            dudt = - decayRate * u0 + heatRate * heat
            return u0 + dudt * dt

        # Integrate the ODEs.
        self.persistedHeat = u = [np.copy(self.heatSources[0])]
        for heat in self.heatSources[1:]:
            u.append(decay(u[-1], heat))

        # Generate a video scaled to the maximum value.
        if outVidPath is None:
            outVidPath = 'doc/%s-heatDecay.mp4' % self.baseLabel

        u = np.stack(u)
        u /= u.max()
        u *= 255
        u = u.astype('uint8')
        return saveVideo(u, outVidPath)


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

