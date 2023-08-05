import cv2
import numpy as np


edge = cv2.imread('edge.webp')
edge_avg = cv2.blur(edge, (9,9))
edge_gaussian = cv2.GaussianBlur(edge, (9,9), cv2.BORDER_DEFAULT)
edge_median = cv2.medianBlur(edge, 9)
edge_bilateral = cv2.bilateralFilter(edge, 9, 5000, 5000)
edge_pyr = cv2.pyrUp(cv2.pyrDown(edge))

cv2.imwrite('canny_pyr.png' ,cv2.Canny(edge_pyr, 100, 300))
cv2.imwrite('canny_pyr2.png' ,cv2.Canny(edge_pyr, 50, 400))
cv2.imwrite( 'canny1.png' ,cv2.Canny(edge, 50, 150))
cv2.imwrite('canny2.png' ,cv2.Canny(edge, 500, 1000))
cv2.imwrite('canny_avg.png' , cv2.Canny(edge_avg, 50, 150))
cv2.imwrite('canny_gaussian.png' ,cv2.Canny(edge_gaussian, 50, 150))
cv2.imwrite('canny2_med.png' ,cv2.Canny(edge_median, 50, 150))
cv2.imwrite('canny2_bi.png' ,cv2.Canny(edge_bilateral, 50, 150))