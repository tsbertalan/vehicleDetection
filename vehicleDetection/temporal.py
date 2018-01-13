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

def cooling(heatSources, heatTransferCoefficient=1, heatCapacity=6, dt=1, powerFactor=10, coolingRate=64, firstTemps=None):
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
        How much thermal inertia the pixels have; units are J/K
    dt : float
        Duration of each step in the simulation; units are s, but not actual video s.
    powerFactor : float
        How much power each detection has; units W/detection.
    coolingRate : float
        Static cooling; units W.
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
        dudt = (power * powerFactor - heatTransferCoefficient * u0 - coolingRate) / heatCapacity
        return u0 + dudt * dt

    # Integrate the ODEs.
    if firstTemps is None:
        firstTemps = np.zeros_like(heatSources[0])
    u = [firstTemps]
    for heat in heatSources:
        u.append(decay(u[-1], heat))
        u[-1][u[-1] < 0] = 0

    u = np.stack(u)
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
            heatingPower = bboxes2power(frame.shape[:2], rawBboxes)
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

    def persist(self, Tthresh=8):
        # Post-process the heat sources to get persistent bounding boxes.self.labels = []
        self.temperatures = cooling(self.heatSources, **self.coolingKwargs)[1:]

        def getBboxes(T):
            T = np.copy(T)
            T[T < Tthresh] = 0
            return vehicleDetection.drawing.labeledBboxes(label(T))

        self.persistentBboxes = [
            getBboxes(temperature)
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
        import cv2
        def f(weights, label, frame, alpha=.8, hot=True):
            cmap = cv2.applyColorMap(
                weights, 
                cv2.COLORMAP_HOT if hot else cv2.COLORMAP_OCEAN
            )
            cmap = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
            return wt(cv2.addWeighted(cmap, alpha, frame, .8, 0), label)

        def genVidFrames():
            for frame, segments in zip(self.inputFrames, zip(*framesToConcatenate)):
                rawBox, power, temperature, persistentBbox = segments
                tstr = 'temp (Tmax=%.2g)' % temperature.max()
                vf = np.vstack([
                    np.hstack([
                        wt(rawBox, 'raw detections'), 
                        f(power, 'heating power', frame, hot=False)
                        ]),
                    np.hstack([
                        f(temperature, tstr, frame), 
                        wt(persistentBbox, 'persistent thresholded components')
                    ]),
                ])
                self.vidFrame = vf
                yield vf

        return saveVideo(
            genVidFrames(),
            outVidPath
        )
