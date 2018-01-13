from scipy.ndimage.measurements import label
import vehicleDetection
from vehicleDetection.videoUtils import *
import os


def intApproxSqrt(n):
    i = 1
    while i <= n:
        pass



def tileImages(*frames):
    pass
    

def bboxes2power(imageShape, bboxes):
    power = np.zeros(imageShape)
    for box in bboxes:
        power[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return power

def cooling(heatSources, heatTransferCoefficient=9, heatCapacity=10, dt=1, firstTemps=None):
    """Model exponential heat decay with source injections.

    Model heat decay at each pixel using
        dudt = (heatTransferCoefficient * A * (0 - u(t)) + h(t)) / heatCapacity
    where A is assumed to be 1 m**2

    Parameters
    ----------
    heatSources : list of ndarray
        Heat source images, units are W.
    heatTransferCoefficient : float
        Rate at which heat leaks from each pixel; units are W/m**2/K
    heatCapacity : float
        How much thermal inertia the pixels have relative to
        heating from detections; units are J/K
    dt : float
        Duration of each step in the simulation; units are s, but not actual video s.
    firstTemps : ndarray
        Initial condition for temperature field, units are K.
        Defaults to zeros of same shape as first heatSources array.

    Returns
    -------
    u : list of ndarray
        List of temperature arrays of length 1+len(heatSources); units are K.
    """
    def decay(u0, power):
        """Euler time stepper/flow map"""
        dudt = (power - heatTransferCoefficient * u0) / heatCapacity
        return u0 + dudt * dt

    # Integrate the ODEs.
    if firstTemps is None:
        firstTemps = np.zeros_like(heatSources[0])
    u = [firstTemps]
    for heat in heatSources:
        u.append(decay(u[-1], heat))

    u = np.stack(u)
    u /= u.max()
    u *= 255
    u = u.astype('uint8')
    return u


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
            rawBboxes = detector.rawDetect(frame)

            # Sum raw bounding boxes to get heating power per pixel.
            heatingPower = bboxes2power(frame.shape, rawBboxes)
            assert (heatingPower >= 0).all()
            assert heatingPower.dtype == float
            # heatingPower /= heatingPower.max()
            # heatingPower *= 255
            # heatingPower = heatingPower.astype('uint8')

            return rawBboxes, heatingPower

        self.heatSources = []
        self.rawBboxes = []
        for frame in tqdm.tqdm_notebook(self.inputFrames, desc='processing', unit='frame'):
            rawBboxes, power = generate(frame)
            self.heatSources.append(power)
            self.rawBboxes.append(rawBboxes)

        self.persist()

    def persist(self):
        # Post-process the heat sources to get persistent bounding boxes.self.labels = []
        self.temperatures = cooling(self.heatSources, **self.coolingKwargs)[1:]
        self.persistentBboxes = [
            vehicleDetection.drawing.labeledBboxes(label(temperature))
            for temperature in tqdm.tqdm_notebook(
                self.temperatures,
                desc='persist bboxes', unit='frame'
            )
        ]

    def video(self, label=None, outVidPath=None):

        # Assemble output path.
        if outVidPath is None:
            outVidPath = 'doc/%s-detected.mp4' % (
                self.baseLabel,
            )
        if label is not None:
            ext = outVidPath[-4:]
            outVidPath.replace(ext, label + ext)

        def rescaleField(mats):
            m = np.stack(mats)            
            m -= m.min()
            m = m / float(m.max())
            m *= 255.
            return m.astype('uint8')

        framesToConcatenate = [
            #self.inputFrames,
            [
                vehicleDetection.drawing.drawBboxes(frame, bbox, color=(255, 0, 0))
                for (frame, bbox) in zip(self.inputFrames, self.rawBboxes)
            ],
            rescaleField(self.heatSources),
            rescaleField(self.temperatures),
            [
                vehicleDetection.drawing.drawBboxes(frame, bbox, color=(255, 0, 255))
                for (frame, bbox) in zip(self.inputFrames, self.persistentBboxes)
            ]
        ]

        from vehicleDetection.drawing import writeText as wt
        return saveVideo(
            (
                np.vstack([
                    np.hstack([wt(rawBox, 'raw'), wt(power, 'power')]),
                    np.hstack([wt(temperature, 'temp'), wt(persistentBbox, 'persit')]),
                ])
                for (
                    rawBox, 
                    power,
                    temperature,
                    persistentBbox
                    )
                in zip(*framesToConcatenate)
            ),
            outVidPath
        )
