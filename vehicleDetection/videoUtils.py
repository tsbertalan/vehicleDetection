import os

import numpy as np
import tqdm

import time
import skvideo.io
from IPython.display import HTML

def loadVideo(
    fpath='../../../data/vehicleDetection/test_video.mp4', 
    maxFrames=None, 
    pbar=True
    ):
    reader = skvideo.io.FFmpegReader(fpath)
    frames = []
    actualMaxFrames = reader.inputframenum if maxFrames is None else maxFrames
    if pbar:
        bar = tqdm.tqdm_notebook(
            total=actualMaxFrames,
            desc='%s' % os.path.basename(fpath),
        )
        update = bar.update
    else:
        update = lambda : None

    for frame in reader.nextFrame():
        if len(frames) == actualMaxFrames:
            break
        update()
        frames.append(frame)
    return frames


def fig2img(fig):
    """Render a Matplotlib figure to an image; good for simple video-making."""
    # stackoverflow.com/questions/35355930
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    canvas = FigureCanvas(fig)
    ax = fig.gca()
    canvas.draw()       # draw the canvas, cache the renderer
    width, _ = fig.get_size_inches() * fig.get_dpi()
    width = int(width)
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(-1, width, 3)
    return image


def saveVideo(frames, fpath, **tqdmKw):
    """Save a collection of images to a video file. I've tried .mp4 extensions."""
    if tqdmKw.pop('pbar', True):
        tqdmKw.setdefault('desc', os.path.basename(fpath))
        tqdmKw.setdefault('unit', 'frame')
        pbar = tqdm.tqdm_notebook
    else:
        pbar = lambda x, **kw: x
    writer = skvideo.io.FFmpegWriter(fpath)
    for frame in pbar(frames, **tqdmKw):
        writer.writeFrame(frame)
    writer.close()
    return showAsHTML(fpath)


def showAsHTML(fpath):
    """Display a video as a Jupyter HTML widget.

    Use a relative path that is accessible via the jupyter notebook webserver.
    """
    # Add a time argument to suggest that chrome shouldn't cache the video.
    t = time.time()

    # Display images with <image>.
    for ext in '.png', '.gif', '.jpg', '.jpeg':
        if fpath.lower().endswith(ext):
            if fpath.lower().endswith('.gif'):
                return jupyterTools.GIFforLatex(fpath)
            else:
                return HTML("""<image src="%s?time=%s" />""" % (fpath, t))

    # Displaly videos with <video>.
    return HTML("""
    <video width=100%% controls loop>
      <source src="%s?time=%s" type="video/mp4">
    </video>
    """ % (fpath, t))


