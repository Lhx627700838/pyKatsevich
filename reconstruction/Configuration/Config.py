"""
defines parameters for scanner and image model
"""
import numpy as np

source = 'Naeotom' #'drr' or 'sim'
drrsize = [256, 128]
# x-ray source parameters
SAD = 610 #927 # source-axis distance in mm
sourceAxisAngle = 90 # degree angle between source-ISO line and z-axis of the imaging system

# flat panel detector parameters
SDD = 1113 #1274
detectorDimension = [256, 144] # in x and z direction (must be at least 2 in each dimension) should be same as drr if choose 'drr' 
detectorResolution = [0.4, 0.4] # mm
detectorOffset = [0.5, 0.5]  # mm, center offset in x and z if choose drr, set [0.5 ,0.5]
detectorTiltAngle = 0 # additional tilt degree angle of the detector panel
detectorAxisAngle = 0 # degree angle between panel-ISO line and z-axis of the imaging system



# projection parameters
nView = 2048 # number of axial angles both for sim and drr
evenlyDistributedView = True
viewSet = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340] # set of axial angles if not axial angles are not evenly distributed
rayTracingMethod = 'distance' # 'distance': length of intersection; 'sampling: sample point on source-detector line
rayTracingSampleInterval = 1 # sampling distance in mm for ray tracing
simulateProjectionMethod = 'forward' # 'analytic': calc line integral from analytic phantom; 'forward': forward projection from digital CT data matrix
projectorPlatform = 'cuda' # option: c, python

# helix parameters
pitch = -46 # unit in mm
angles_range = 34.03 # unit in rad
views_per_turn = nView/(angles_range/(2*np.pi))
deltaZ = pitch/views_per_turn
total_angle = (angles_range/(2*np.pi))*360
deltaLamda = total_angle/nView

# reconstruction parameters
reconDiameter = 5000 # diameter of in-plane reconstruction in mm
reconMethod = 'fbp' # supported methods: fbp, mlem
fbpFilter = 'hamming' # supported filters: ramp, shepp-logan, cosine, hamming and hann
fbpCutoffFreq = 0.8 # cutoff frequency (relative to max freq) during filtering in FBP
mlemIteration = 30 # number of iterations in MLEM reconstruction
waterMu = 0.0196 # the attenuation of water that will be used in HU conversion
reconImageDimension = [128, 128, 256] # number of voxels in x, y and z dimensions
reconImageResolution = [0.5, 0.5, 0.5] # image resolution in mm 
reconImageOffset = [0, 0, -128] # mm, recon center offset

# other parameters
patientPosition = 'HFS'
multiprocessing = False  # multiprocess in forward and back projection

# simulation parameters
simulationPhantom = 'CtSeg1' # options: head, shepplogan3d, or CtSeg
simImageDimension = reconImageDimension # number of voxels in x, y and z dimensions
simImageResolution = reconImageResolution # image resolution in mm
simImageOffset = reconImageOffset # mm, recon center offset