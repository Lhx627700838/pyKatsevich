"""
save image to DICOM format
"""
import os
import shutil
import numpy as np


def SaveRaw(image, outputDir,filename):

    dirScript = os.path.dirname(os.path.realpath(__file__))
    dirScript = dirScript.replace('/Utils', '')

    imageDIR = dirScript + '/output/' + outputDir
    if os.path.isdir(imageDIR):
        shutil.rmtree(imageDIR)
    os.makedirs(imageDIR, exist_ok = True)

    filenameFull = imageDIR + '/' + filename
    with open(filenameFull, 'wb') as fl:
        for iz in range(image.data.shape[2]):
            fl.write(np.float32(image.data[:, :, iz]))



