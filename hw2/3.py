import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage.draw import ellipse_perimeter
from skimage import color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse, pyramids

#this function perform hough transformation
#for detecting ellipses and then draw ellipse on 
#input image
def ellipse(image_rgb, edges, count=1 , acc = 1 , thr = 5):
    #performin hough transformation
    result = hough_ellipse(edges ,accuracy=acc , threshold=thr)#, min_size=5, max_size=35)
    #sorting based on accumulator
    result.sort(order='accumulator')
    # After finding all the possible ellipses in the the image using the Hough transform,
    # they are sorted based on multiple criterion for the most plausible ellipse to be found. These criterion are: 
    # * Accumulation
    # * Deviance from the 90°
    # * Deviance from ideal face height and width
    # * Deviance from the ideal area
    result = sorted(list(result), 
                    key=lambda x: \
                    -x[0]/20 + abs(x[4]-1.5*x[3])/3 + abs(x[5]-np.pi/2)/.6 + \
                    (abs(image_rgb.shape[1]/5-x[4]) + abs(image_rgb.shape[0]/5-x[3]))/3 #+ \
                    #abs((image_rgb.shape[0]*image_rgb.shape[1])/(x[3]*x[4]) - 4)
                    )
    
    edges = color.gray2rgb(img_as_ubyte(edges))
    #drawing ellipse
    for res in result[:min(len(result), count)]:
        res = list(res)
        orientation = res[5]
        yc, xc, a, b = [int(round(x)) for x in res[1:5]]
        
        # if .6 < abs(orientation - np.pi/2): continue
        # if not abs(b - 1.5 * a) < 3: continue
        # if not 5 < b < 20 or not 5 < a < 15: continue

        cy, cx = ellipse_perimeter(yc, xc, a, b, orientation, shape=image_rgb.shape)
        image_rgb[cy, cx] = (255, 0, 0)
        edges[cy, cx] = (250, 0, 0)
        
    return image_rgb, edges

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
	
    if ret: #if frame has fecthed correctly    
        #changing frame to rgb format for working with skimage
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #performing gaussian pyramid
        image_rgb = tuple(pyramids.pyramid_gaussian(image, 3, multichannel=1))[-1]
        #turning image to gray scale and performin canny
        image_gray = color.rgb2gray(image_rgb)
        edges = canny(image_gray, sigma=1.5, low_threshold=0.2, high_threshold=0.8)
        #perforimng ellipse detection
        image_rgb, edges = ellipse(image_rgb, edges, acc = 20)
        #changing format to bgr
        final = image_rgb[:,:,::-1]
        final = cv2.resize(final, ( 640 , 480)) 
        cv2.imshow('ellipse',final )

    cv2.imshow('frame', frame)
    
    
    if cv2.waitKey(1) == 27: #waiting 30 miliseconds
        # ESC pressed
        print("Escape hit, closing...")
        break 
#releasing cam
cap.release()
cv2.destroyAllWindows()


#performing all of proccess on last frame for different scales of pyramid
image_rgb = tuple(pyramids.pyramid_gaussian(image, 3, multichannel=1))[-1]
image_gray = color.rgb2gray(image_rgb)
edges = canny(image_gray, sigma=1.5, low_threshold=0.2, high_threshold=0.8)
    
image_rgb, edges = ellipse(image_rgb, edges, acc = 20)

fig2, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, figsize=(12, 4))
ax1.imshow(image);
ax2.imshow(image_rgb);
ax3.imshow(edges);
plt.show()
plt.imsave('3pyr.png' ,edges)

image_rgb = tuple(pyramids.pyramid_gaussian(image, 2, multichannel=1))[-1]
image_gray = color.rgb2gray(image_rgb)
edges = canny(image_gray, sigma=1., low_threshold=0.2, high_threshold=0.8)
    
image_rgb, edges = ellipse(image_rgb, edges , acc = 30 , thr = 10)

fig2, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, figsize=(12, 4))
ax1.imshow(image);
ax2.imshow(image_rgb);
ax3.imshow(edges);
plt.show()
plt.imsave('2pyr.png' ,edges)

