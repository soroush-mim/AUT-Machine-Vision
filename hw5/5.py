import cv2
import numpy as np
from scipy.spatial import distance

def draw_flow(img,flow,step=32 , color = (0,0,255)):
    #a function for drawing flow vectors on the image
    h,w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int) 
    fx,fy = flow[y,x].T

    # create line endpoints
    lines = np.vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
    lines = np.int32(lines)

    for (x1,y1),(x2,y2) in lines:
        #flow vectors are bigger than a treshold
        #then we will draw them
        if distance.euclidean((x1,y1),(x2,y2)) > 3:
            img = cv2.line(img,(x1,y1),(x2,y2),color,2)
            img = cv2.circle(img,(x1,y1),1,color, 2)
    return img
    
########################################################################
#initialize parameters
# a list for keeping last 3 frames
images = []
#reading frames from webcam and doing the proccess
cam=cv2.VideoCapture(0)
#creating objects of videoWriter for writing video
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('5.avi',fourcc, 6,(int(cam.get(3)),int(cam.get(4))))
heatmap_out = cv2.VideoWriter('heatmap.avi',fourcc, 6,(int(cam.get(3)),int(cam.get(4))))
########################################################################
ret, QueryImgBGR=cam.read()
QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY) #turning frame to gray
images.append(QueryImg)
heatmap = np.zeros_like(QueryImgBGR)
heatmap[..., 1] = 255

while True:
    #reading frames
    ret, QueryImgBGR=cam.read()
    QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY) #turning frame to gray

    if ret:
        images.append(QueryImg)
        
    if len(images) > 2: #skiping till we have seen more than 2 frames
        
        # calculate optical flow
        flow1 = cv2.calcOpticalFlowFarneback(images[0], images[1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow2 = cv2.calcOpticalFlowFarneback(images[1], images[2], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        #drawing heatmap
        mag, ang = cv2.cartToPolar(flow2[..., 0], flow2[..., 1])
        heatmap[..., 0] = ang*180/np.pi/2
        heatmap[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(heatmap, cv2.COLOR_HSV2BGR)
        bgr = cv2.addWeighted(QueryImgBGR,.6,bgr,.5,0)
        #drawing flows
        draw_flow(QueryImgBGR, flow1 , color = (0,0,255))
        draw_flow(QueryImgBGR, flow2 , color = (255,0,0))

        cv2.imshow('frame', QueryImgBGR)
        cv2.imshow('heatmap', bgr)
        out.write(QueryImgBGR)
        heatmap_out.write(bgr)
        images.pop(0)

    if cv2.waitKey(10)==27:
        break

out.release()
cam.release()
cv2.destroyAllWindows()