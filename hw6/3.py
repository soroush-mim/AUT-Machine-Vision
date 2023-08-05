import cv2
import numpy as np
from matplotlib import pyplot as plt

def dist(l_patch, r_patch):
    return np.sum(np.power(l_patch - r_patch , 2))

def disparity(imgL, imgR, patch_size , max_sv, contrast =10):
    """generate disparity map based on left and right images

    Args:
        imgL ([numpy 2D array]): [left image (gray)]
        imgR ([numpy 2D array]): [right image (gray)]
        patch_size ([int]): [defines size of patch]
        max_sv ([int]): [maximum size for searching in right image on the same line]
        contrast (int, optional): [constant value for better visualization]. Defaults to 10.

    Returns:
        [numpy 2D array]: [disparity map with the same size as the input photos]
    """    
    H,W = imgL.shape
    dis_map = np.zeros_like(imgL)
    #doing process for each pixel
    for row in range(patch_size//2, H-patch_size//2):
        for col in range(patch_size//2, W-patch_size//2):
            #selecting patch in left image
            l_patch = imgL[row - patch_size//2:row + patch_size//2 +1 ,col - patch_size//2:col + patch_size//2 +1]
            min_dist = 1000
            match = 0
            for i in range(max_sv):
                if(col - patch_size//2 - i > -1):
                    #selecting patch in right image
                    r_patch = imgR[row - patch_size//2:row + patch_size//2 +1 ,col - patch_size//2 - i :col + patch_size//2 +1-i]
                    new_dist = dist(l_patch,r_patch)
                    if new_dist < min_dist:
                        min_dist = new_dist
                        match = i
            dis_map[row,col] = match *contrast if match*contrast < 255 else 255
        
    return dis_map

            
#reading images
imgL = cv2.imread('left.png',0)
imgR = cv2.imread('right.png',0)
#downsampling
imgL = cv2.resize(imgL, None, fx=1/2, fy=1/2, interpolation=cv2.INTER_AREA)
imgR = cv2.resize(imgR, None, fx=1/2, fy=1/2, interpolation=cv2.INTER_AREA)
#denoise
imgL = cv2.GaussianBlur(imgL, (3,3), cv2.BORDER_DEFAULT)
imgR = cv2.GaussianBlur(imgR, (3,3), cv2.BORDER_DEFAULT)

dis_map = disparity(imgL,imgR,5,40,4)

# cv2.imwrite('dis.png',dis_map)
plt.imsave('dis.png',dis_map)

for i in [5,7,9]:
    img =  cv2.medianBlur(dis_map,i)
    # cv2.imwrite('dis_med'+str(i)+'.png',img)
    plt.imsave('dis_med'+str(i)+'.png',img)

img = dis_map
for i in range(4):
    img = cv2.medianBlur(img,5)

# cv2.imwrite('dis_med_itr.png',img)
plt.imsave('dis_med_itr.png',img)