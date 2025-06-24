"""
defines object-system transformation for 3d points
"""
import math
import numpy as np

def TranslateSytem2ObjectCoordinate(pt, origin, fTheta, fPhi):

    fTheta = fTheta * np.pi/180.0
    fPhi = fPhi * np.pi/180.0
    ptnew = np.zeros(3)

    ptnew[0] = ((pt[0] - origin[0]) * math.cos(fTheta) + (pt[1] - origin[1]) * math.sin(fTheta)) * math.cos(fPhi) + (pt[2] - origin[2]) * math.sin(fPhi)
    ptnew[1] = -(pt[0] - origin[0]) * math.sin(fTheta) + (pt[1] - origin[1]) * math.cos(fTheta)
    ptnew[2] = -((pt[0] - origin[0]) * math.cos(fTheta) + (pt[1] - origin[1]) * math.sin(fTheta)) * math.sin(fPhi) + (pt[2] - origin[2]) * math.cos(fPhi)

    return ptnew


def TranslateObject2SystemCoordinate(pt, origin, fTheta, fPhi):

   fTheta = fTheta * np.pi / 180.0
   fPhi = fPhi * np.pi / 180.0
   ptnew = np.zeros(3)
   ptnew[0] = (pt[0] * math.cos(fPhi) - pt[2] * math.sin(fPhi)) * math.cos(fTheta) - pt[1] * math.sin(fTheta) + origin[0]
   ptnew[1] = (pt[0] * math.cos(fPhi) - pt[2] * math.sin(fPhi)) * math.sin(fTheta) + pt[1] * math.cos(fTheta) + origin[1]
   ptnew[2] = pt[0] * math.sin(fPhi) + pt[2] * math.cos(fPhi) + origin[2]

   return ptnew
