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

def cooling(heatSources, heatTransferCoefficient=1, heatCapacity=20, dt=1, powerFactor=20, coolingRate=10, firstTemps=None):
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
        # dudt = power - u0.astype(bool).astype(int) * 3
        # return u0 + dudt * dt
        dudt = (power * powerFactor - heatTransferCoefficient * u0 - coolingRate) / heatCapacity
        return u0 + dudt * dt

    # Integrate the ODEs.
    if firstTemps is None:
        firstTemps = np.zeros_like(heatSources[0])
    u = [firstTemps]
    for heat in tqdm.tqdm_notebook(heatSources, desc='cooling', unit='frame'):
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
        self.detectorLabel = None

    def go(self, detector, **skw):

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

        self.detectorLabel = detector.label
        self.save(**skw)

    def persist(self, Tthresh=20):
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

    def save(self, fpath='/home/tsbertalan/data/vehicleDetection/detections.h5', addLabel=True, guard=None):
        import h5py
        if addLabel:
            fpath = fpath[:-3] + '-' + self.baseLabel
            if self.detectorLabel is not None:
                fpath += '-' + self.detectorLabel
            fpath += '.h5'
        print('Saving to %s ...' % fpath, end=' ')

        f = h5py.File(fpath, 'w')
        try:
            # Save the heat sources (several gigabytes).
            data = self.heatSources[:guard]
            out = f.create_dataset(
                'power',
                (len(data), *data[0].shape),
                dtype=data[0].dtype
            )
            for i, x in enumerate(tqdm.tqdm_notebook(data, desc='power images', unit='frame')):
                out[i] = x

            # Save the raw bounding boxes (several kilobytes).
            data = self.rawBboxes[:guard]
            dt = h5py.special_dtype(vlen=np.array(data[0]).dtype)
            out = f.create_dataset('rawBboxes', (len(data),), dtype=dt)
            for i, x in enumerate(data):
                x = np.array(x).ravel()
                out[i] = x
        except:
            pass
        f.close()
        
        print('done.')

    def load(self, fpath='/home/tsbertalan/data/vehicleDetection/detections.h5'):
        import h5py
        if hasattr(self, '_loadedFile'): self._loadedFile.close()
        self._loadedFile = f = h5py.File(fpath, 'r')
        # try:
        # heatSources = [hs for hs in f['power']]
        # self.heatSources = heatSources
        self.heatSources = f['power']
        bboxes = [[tuple(map(tuple, b)) for b in bb.reshape((-1, 2, 2))] for bb in f['rawBboxes']]
        self.rawBboxes = bboxes
        #     f.close()
        # except:
        #     f.close()
            
    def video(self, label=None, outVidPath=None):
        # Assemble output path.
        if outVidPath is None:
            outVidPath = 'doc/%s-detected.mp4' % (
                self.baseLabel,
            )
        l = '' if label is None else label
        if self.detectorLabel is not None:
            l += '-' + self.detectorLabel
        if len(l) > 0:
            ext = outVidPath[-4:]
            outVidPath = outVidPath.replace(ext, l + ext)
        print('Generating video %s.' % outVidPath)

        highPower = max([x.max() for x in tqdm.tqdm_notebook(self.heatSources, desc='hmax?')])
        highTemp = max([x.max() for x in self.temperatures])

        def genVidFrames():
            print('Generating %d video frames.' % len(self.inputFrames))
            """Assemble the video frames in a generator to conserve memory."""
            from vehicleDetection.drawing import writeText as tlabel
            import cv2

            for i in range(len(self.inputFrames)):
                frame = self.inputFrames[i]

                def overlay(weights, alpha=.8, hot=True):
                    """Overlay weights on current frame."""
                    cmap = cv2.applyColorMap(
                        weights, 
                        cv2.COLORMAP_HOT if hot else cv2.COLORMAP_OCEAN
                    )
                    cmap = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
                    return cv2.addWeighted(cmap, alpha, frame, .8, 0)

                rawBox = vehicleDetection.drawing.drawBboxes(
                    frame, 
                    self.rawBboxes[i],
                    color=(255, 0, 0)
                )
                power = (self.heatSources[i] / highPower * 255).astype('uint8')
                temperature = (self.temperatures[i] / highTemp * 255).astype('uint8')
                persistentBbox = vehicleDetection.drawing.drawBboxes(
                    frame,
                    self.persistentBboxes[i],
                    color=(255, 0, 255)
                )
                tstr = 'temp (Tmax=%.3g [K])' % temperature.max()
                vf = np.vstack([
                    np.hstack([
                        tlabel(rawBox, 'raw detections'), 
                        tlabel(overlay(power, hot=False), 'heating power')
                        ]),
                    np.hstack([
                        tlabel(overlay(temperature), tstr), 
                        tlabel(persistentBbox, 'persistent thresholded components')
                    ]),
                ])
                self.vidFrame = vf
                yield vf

        return saveVideo(
            genVidFrames(),
            outVidPath,
            total=len(self.inputFrames),
        )

    def __del__(self):
        if hasattr(self, '_loadedFile'): self._loadedFile.close()