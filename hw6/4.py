import cv2
from matplotlib import pyplot as plt


imgL = cv2.imread('left.png',0)
imgR = cv2.imread('right.png',0)

# The disparity search range [minDisparity, minDisparity+numberOfDisparities].
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=19 )

# filters out areas that don't have enough texture for reliable matching
stereo.setTextureThreshold(5)

# accept the computed disparity d* only ifSAD(d) >= SAD(d*)(1 + uniquenessRatio/100.)
#  for any d != d+/-1 within the search range
stereo.setUniquenessRatio(5)

# Block-based matchers often produce "speckles" near the boundaries of objects, where the matching window
#  catches the foreground on one side and the background on the other. In this scene it appears that the
#   matcher is also finding small spurious matches in the projected texture on the table. To get rid of these
#    artifacts we post-process the disparity image with a speckle filter controlled by the speckle_size and 
#    speckle_range parameters. speckle_size is the number of pixels below which a disparity blob is dismissed
#     as "speckle."
stereo.setSpeckleWindowSize(0)

# prefilter_size and prefilter_cap: The pre-filtering phase, which normalizes image brightness 
# and enhances texture in preparation for block matching. Normally you should not need to adjust these.
stereo.setPreFilterSize(5)
disparity = stereo.compute(imgL,imgR)
plt.imsave('stereo-bm.png',disparity)







        

