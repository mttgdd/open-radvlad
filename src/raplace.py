import numpy as np
from skimage.transform import radon
from skimage.transform import rescale
from tqdm import tqdm
    
def sino_fft(x, cfg):
    theta = np.arange(0,180)
    R = radon(x, theta=theta)
    R /= np.max(R)
    R = R.astype(np.float64)
    R = rescale(R, scale=cfg['scale'])
    R = np.abs(np.fft.fft(R, axis=0))
    R = R[:int(R.shape[0]/2),:]
    R = (R-np.mean(R))/np.std(R)
    return R

def do_sino_fft(cfg, dataset):
    sino_ffts = []
    for img in tqdm(dataset, total=len(dataset), desc=f"Collecting Sino FFTs"):
        sino_ffts.append(sino_fft(img, cfg))
    return np.array(sino_ffts)

def circular_xcorr(Mq, Mi):
    Fq = np.fft.fft(Mq, axis=0)
    Fn = np.fft.fft(Mi, axis=0)
    corrmap_2d = np.fft.ifft(Fq*np.conj(Fn), axis=0)
    corrmap = np.sum(corrmap_2d,axis=-1)
    maxval = np.max(corrmap)
    return maxval

def max_circular_xcorr(Mq, Mi):
    mCauto = circular_xcorr(Mq, Mq)
    mCqi = circular_xcorr(Mq, Mi)
    return np.abs(mCauto-mCqi)
