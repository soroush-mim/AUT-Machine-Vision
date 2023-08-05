import cv2
import numpy as np

#reading images
hist1 = cv2.imread('Hist1.webp')
hist2 = cv2.imread('Hist2.webp')
#converting to grayscale
hist1_gray = cv2.cvtColor(hist1, cv2.COLOR_BGR2GRAY)
hist2_gray = cv2.cvtColor(hist2, cv2.COLOR_BGR2GRAY)
#showing images
for img in [hist1, hist1_gray, hist2, hist2_gray]:
    cv2.imshow('image',img)
    cv2.waitKey(0)
cv2.destroyAllWindows()
#saving grayscale images
cv2.imwrite('hist1_gray.jpg' , hist1_gray)
cv2.imwrite('hist2_gray.jpg', hist2_gray)