import cv2
import numpy as np

#initialize parameters
# params for ShiTomasi corner detection function
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# a list for keeping last 3 frames
images = []
# Parameters for lucas kanade optical flow function
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
########################################################################
#reading frames from webcam and doing the proccess
cam=cv2.VideoCapture(0)
#creating objects of videoWriter for writing video
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('2.avi',fourcc, 10,(int(cam.get(3)),int(cam.get(4))))

# Taking first frame and find corners in it
ret, first_frame = cam.read()
old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
images.append(old_gray)
p0 = cv2.goodFeaturesToTrack(images[0], mask = None, **feature_params)
########################################################################
while True:

    #reading frames
    ret, QueryImgBGR=cam.read()
    QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY) #turning frame to gray

    if ret:
        images.append(QueryImg)
    
    if len(images) > 2: #skiping till we have seen more than 2 frames
        
        # calculating optical flows
        p1, st1, er1 = cv2.calcOpticalFlowPyrLK(images[0], images[1], p0, None, **lk_params)
        p2, st2, er2 = cv2.calcOpticalFlowPyrLK(images[1], images[2], p1, None, **lk_params)

        # Selecting good points
        if p1 is not None:
            good_new1 = p1[st1==1]
            good_old1 = p0[st1==1]

        if p2 is not None:
            good_new2 = p2[st2==1]
            good_old2 = p1[st2==1]

        # draw the tracks
        for (new, old) in zip(good_new1, good_old1):
            a, b = new.ravel()
            c, d = old.ravel()
            QueryImgBGR = cv2.line(QueryImgBGR, (int(a), int(b)), (int(c), int(d)), (0,0,255), 2)
            QueryImgBGR = cv2.circle(QueryImgBGR, (int(a), int(b)), 5,(0,0,255) , -1)

        for (new, old) in zip(good_new2, good_old2):
            a, b = new.ravel()
            c, d = old.ravel()
            QueryImgBGR = cv2.line(QueryImgBGR, (int(a), int(b)), (int(c), int(d)), (255,0,0), 2)
            QueryImgBGR = cv2.circle(QueryImgBGR, (int(a), int(b)), 5, (255,0,0), -1)

        images.pop(0)
        
        p0 = good_new1.reshape(-1, 1, 2)
        p1 = good_new2.reshape(-1, 1, 2)

        cv2.imshow('frame', QueryImgBGR)
        out.write(QueryImgBGR)

    if cv2.waitKey(10)==27:
        break

out.release()
cam.release()
cv2.destroyAllWindows()