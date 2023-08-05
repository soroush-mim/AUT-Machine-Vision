import cv2
import numpy as np

hist1 = cv2.imread('Hist1.webp')
#creating a mask for white color using trasholding 
#in (200 , 255) range on colors
mask = cv2.inRange(hist1, np.array([200, 200, 200], dtype="uint8"), np.array([255, 255, 255], dtype="uint8"))

mask2 = np.zeros_like(hist1)
mask2[:, :, 0] = mask
mask2[:, :, 1] = mask
mask2[:, :, 2] = mask
# hist and (not mask) -> turns white color to black
hist1 = cv2.bitwise_and(hist1, 255 - mask2)
#grayscale transform
hist1_gray = cv2.cvtColor(hist1, cv2.COLOR_BGR2GRAY)
#inverting grayscale photo
hist1_gray_inverted = cv2.bitwise_not(hist1_gray)

#converting color space to HSV
hist1_hsv = cv2.cvtColor(hist1 , cv2.COLOR_BGR2HSV)
#creating a mask for yellow color
mask3 = cv2.inRange(hist1_hsv, (10,0,0), (35, 255, 255))
# not (hist1gray and yellow mask) -> first turns yellow color
# to white then invert the photo
hist1_masked = cv2.bitwise_not(cv2.bitwise_and(hist1_gray , mask3))
#inverting yellow mask
mask3_inv = cv2.bitwise_not(mask3)

cv2.imwrite('hist1_black_background.png' , hist1)
cv2.imwrite('hist1_gray_inv.png' , hist1_gray_inverted)
cv2.imwrite('hist1_mask3_inv.png' , mask3_inv)
cv2.imwrite('hist1_masked_inv.png' , hist1_masked)