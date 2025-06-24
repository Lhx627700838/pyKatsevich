"""
save image to DICOM format
"""
import os
from Configuration import Config
import datetime
import shutil
from pydicom import dcmread
from pydicom.uid import generate_uid
from pydicom.data import get_testdata_file


def SaveDICOM(image, outputDir):

    dirScript = os.path.dirname(os.path.realpath(__file__))
    dirScript = dirScript.replace('/Utils', '')

    imageDIR = dirScript + '/output/' + outputDir
    if os.path.isdir(imageDIR):
        shutil.rmtree(imageDIR)
    os.makedirs(imageDIR, exist_ok = True)

    if image.data.ndim == 2:
        nSlice = 1
    else:
        nSlice = image.data.shape[2]

    # creating DICOM template
    templateDICOMPath = get_testdata_file("CT_small.dcm")
    templateDICOM = dcmread(templateDICOMPath)
    UID = generate_uid()
    dateTimeNow = datetime.datetime.now()
    dateStr = str(dateTimeNow.date()).replace('-', '')
    timeStr = str(dateTimeNow.time()).split('.')[0].replace(':', '')

    for iSlice in range(nSlice):
        imageSlice = image.data[:, :, iSlice].astype('uint16')
        templateDICOM.PixelData = imageSlice.tobytes()
        templateDICOM[0x0028, 0x0010].value = image.dimension[0]
        templateDICOM[0x0028, 0x0011].value = image.dimension[1]
        templateDICOM[0x0020, 0x0013].value = iSlice + 1
        templateDICOM[0x0020, 0x1041].value = image.voxelCoordinate[2, 0, 0, iSlice]
        templateDICOM[0x0008, 0x0018].value = UID
        templateDICOM[0x0008, 0x0012].value = dateStr
        templateDICOM[0x0008, 0x0020].value = dateStr
        templateDICOM[0x0008, 0x0021].value = dateStr
        templateDICOM[0x0008, 0x0022].value = dateStr
        templateDICOM[0x0008, 0x0023].value = dateStr
        templateDICOM[0x0008, 0x0013].value = timeStr
        templateDICOM[0x0008, 0x0030].value = timeStr
        templateDICOM[0x0008, 0x0031].value = timeStr
        templateDICOM[0x0008, 0x0032].value = timeStr
        templateDICOM[0x0008, 0x0033].value = timeStr
        templateDICOM[0x0018, 0x1100].value = Config.reconDiameter
        templateDICOM[0x0018, 0x1110].value = Config.SDD
        templateDICOM[0x0018, 0x1111].value = Config.SAD
        templateDICOM[0x0018, 0x5100].value = Config.patientPosition
        templateDICOM[0x0018, 0x0050].value = image.dimension[2]
        templateDICOM[0x0018, 0x0088].value = image.dimension[2]

        filenameDICOM = imageDIR + '/' + str(iSlice+1) + '.dcm'
        templateDICOM.save_as(filenameDICOM)



