"""
filtered back projection
"""
import copy
from Configuration import Config
import numpy as np
from scipy.fftpack import fft, ifft, fftshift, fftfreq
from Projection.BackProjection import BackProjection


def DefineFourierFilter(size, filterName, filterCutoff):
    n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=int),
                        np.arange(size / 2 - 1, 0, -2, dtype=int)))
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2

    # Computing the ramp filter from the fourier transform of its frequency domain representation
    fourierFilter = np.real(fft(f))
    freqPts = fftfreq(size)[1:]
    omega = np.pi * freqPts / filterCutoff
    lpWindow = np.ones(size-1)
    if filterName == "ramp":
        pass
    elif filterName == "shepp-logan":
        # Start from first element to avoid divide by zero
        lpWindow = np.sin(omega) / omega
    elif filterName == "cosine":
        lpWindow = np.cos(omega)
    elif filterName == "hamming":
        lpWindow = 0.54 + 0.46 * np.cos(2*omega)
    elif filterName == "hann":
        lpWindow = 0.5 + 0.5 * np.cos(2*omega)
    elif filterName is None:
        fourierFilter[:] = 1

    for i in range(size-1):
        if freqPts[i] > filterCutoff/2 or freqPts[i] < -filterCutoff/2:
            lpWindow[i] = 0

    fourierFilter[1:] *= lpWindow

    return fourierFilter[:, np.newaxis]


def FilterProjection(projectionData, image, source, detector):

    if Config.evenlyDistributedView:
        nView = Config.nView
        viewSet = np.arange(nView) * (360 / nView)
    else:
        nView = len(Config.viewSet)
        viewSet = Config.viewSet

    if nView != np.shape(projectionData)[2]:
        raise Exception("nView in Config.py different from projection data")

    projectionSizePadded = max(64, int(2 ** np.ceil(np.log2(2 * detector.dimension[0]))))
    padWidth = ((0, projectionSizePadded - detector.dimension[0]), (0, 0))
    filterName = Config.fbpFilter
    filterCutoff = Config.fbpCutoffFreq
    fourierFilter = DefineFourierFilter(projectionSizePadded, filterName, filterCutoff)
    detWidthIso = detector.resolution[0] * detector.SAD/detector.SDD

    filteredProjectionData = np.zeros([detector.dimension[0], detector.dimension[1], nView])
    for iView in range(nView):
        projectionView = projectionData[:, :, iView]
        projectionViewPadded = np.pad(projectionView, padWidth, mode='constant', constant_values=0)
        filteredProjectionData[:, :, iView] = np.real(ifft(fft(projectionViewPadded, axis=0) * fourierFilter, axis=0)[:detector.dimension[0], :]) / detWidthIso

    return filteredProjectionData

def GeometryWeighProjection(projectionData, image, source, detector):

    if Config.evenlyDistributedView:
        nView = Config.nView
        viewSet = np.arange(nView) * (360 / nView)
    else:
        nView = len(Config.viewSet)
        viewSet = Config.viewSet

    if nView != np.shape(projectionData)[2]:
        raise Exception("nView in config.py different from projection data")

    source.ComputeSourceCoordinate(source.SAD, source.sourceAxisAngle, 0 , 0)
    detector.ComputeDetectCoordinate(detector.SAD, detector.SDD, detector.dimension, detector.resolution,
                                     detector.offset, detector.detectorAxisAngle, detector.detectorTiltAngle,
                                     0, 0, 0)
    dx = detector.coordinate[0, :, :] - source.coordinate[0]
    dy = detector.coordinate[1, :, :] - source.coordinate[1]
    dz = detector.coordinate[2, :, :] - source.coordinate[2]
    geoWeight = abs(Config.SDD/np.sqrt(dy*dy + dx*dx + dz*dz))
    for iView in range(nView):
        projectionData[:, :, iView] = projectionData[:, :, iView] * geoWeight

    return projectionData

def RedundancyWeighProjection(projectionData, image, source, detector):

    if Config.evenlyDistributedView:
        nView = Config.nView
        viewSet = np.arange(nView) * (360 / nView)
    else:
        nView = len(Config.viewSet)
        viewSet = Config.viewSet

    if nView != np.shape(projectionData)[2]:
        raise Exception("nView in config.py different from projection data")

    redunWeight = 0.5  # for 360 angles
    for iView in range(nView):
        projectionData[:, :, iView] = projectionData[:, :, iView] * redunWeight

    return projectionData

def FilteredBackprojection(projectionData, image, source, detector):

    reconImage = copy.deepcopy(image)

    # geometry weighting
    # projectionData = GeometryWeighProjection(projectionData, image, source, detector)
    # print('step1')
    # print(np.min(projectionData))
    # redundancy weighting
    # projectionData = RedundancyWeighProjection(projectionData, image, source, detector)
    # print('step2')
    # print(np.min(projectionData))
    # filtering
    # filteredProjectionData = FilterProjection(projectionData, image, source, detector)
    # back projection
    # print('step3')
    # print(np.min(filteredProjectionData))
    reconImage.data = BackProjection(projectionData, image, source, detector)
    reconImage.data = reconImage.data 
    print("")

    return reconImage