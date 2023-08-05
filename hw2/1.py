import cv2
import numpy as np

#reading image
img = cv2.imread('sudoku.jpg')
#making a copy of main image
img_p = img.copy()
#changing image to grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#performing edge detector on image
edges = cv2.Canny(gray,50,150,apertureSize=3)
#performing Probabilistic Hough Transform
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 150, minLineLength=10, maxLineGap=250)
#drawing lines
for x1, y1, x2, y2 in lines[:,0]:
    cv2.line(img_p, (x1, y1), (x2, y2), (0, 0, 255), 3)

#performing Hough Transform
lines = cv2.HoughLines(edges,1,np.pi/180,200)
#drawing lines
for rho,theta in lines[:,0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
#saving images
cv2.imwrite('houghlinesP.jpg',img_p)
cv2.imwrite('houghlines.jpg',img)
