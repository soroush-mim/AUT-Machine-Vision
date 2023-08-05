import cv2
import numpy as np
import matplotlib.pyplot as plt

def center(QueryImg  , detector,freak, bf , trainDesc ):
    """this function calculate sift keypoints and freak discriptors
        for an image then matches these features with keypoints
        of train img and detect the object and return center and border
        of object

    Args:
        QueryImg ([2d np array]): [input img]
        detector ([CV2 detector]): [detctor that we want to use  (here we use SIFT)]
        freak ([cv2 descriptor]): [freak descriptor]
        flann ([cv2.FlannBasedMatcher]): [matcher for mathcing features]
        trainDesc ([cv2 descriptor]): [descriptors of train img]

    Returns:
        return 4 vars
        first one tell us that we have detected object in img
        seconde one is avg point of mathced keypoints
        third one is border of detcted object
        fourth on is avg point of vertixes of border
    """

    #computing keypoints for input img using sift
    queryKP=detector.detect(QueryImg,None)
    #computing descriptors for input img using freak
    queryKP,queryDesc= freak.compute(QueryImg,queryKP)
    #matching calculated descriptors and train img descriptors based on hamming 
    #distance and using brute force
    matches=bf.match(queryDesc,trainDesc)
    #sorting matches based on distances
    matches = sorted(matches, key = lambda x:x.distance)
    goodMatch=matches
    #if we have matched enough keypoints for detection
    if(len(goodMatch)>MIN_MATCH_COUNT):
        
        tp=[] # keeping mathced points in train image
        qp=[] # keeping mathced points in input image
        #keeping 80% of best macthes
        for m in goodMatch[:int(len(goodMatch)*.8)]:
            tp.append(trainKP[m.trainIdx].pt)
            qp.append(queryKP[m.queryIdx].pt)

        tp,qp=np.float32((tp,qp))

        #finding borders of detected object
        H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
        h,w=trainImg.shape
        trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
        queryBorder=cv2.perspectiveTransform(trainBorder,H)
        

        qp=np.float64(qp)
        #avg of matched keypoints
        KP_avg = np.int64(np.sum(qp , axis=0) / qp.shape[0])
        #avg of vertixes of detected border
        border_avg = np.int64(np.sum(queryBorder[0] , axis=0) / queryBorder[0].shape[0])
        print("Match Found")

        return True , KP_avg, queryBorder ,border_avg
        
    else:
        print("Not Enough match found-")
        print(len(goodMatch),MIN_MATCH_COUNT)
        return False , False

########################################################################
#initialize parameters
#minimum match points to accept detection
MIN_MATCH_COUNT=80

#creating SIFT object
detector=cv2.cv2.xfeatures2d.SIFT_create()

#creating freak object
freak = cv2.xfeatures2d.FREAK_create()

#initializing matcher, hence freak is a binary descriptor 
#we will use hamming dist
bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)

# a list for keeping last 3 frames
images = []
########################################################################
#computing sift keypoints and computing freak descriptors using these
#keypoints for train img and plotting them
#i also plotted the avg point of keypoints as a purple circle
#reading train img
trainImg=cv2.imread("train.jpg",0)

#computing sift keypoints
kp = detector.detect(trainImg,None)

#computing freak keypoints and descriptors
trainKP,trainDesc= freak.compute(trainImg,kp)

#drawing keypoints
trainImg1=cv2.drawKeypoints(trainImg,trainKP,None,(255,0,0),4)

#drawing avg of keypoints
points = [i.pt for i in trainKP]
points = np.float64(points)
center_coordinates = np.int64(np.sum(points , axis=0) / points.shape[0])
trainImg1 = cv2.circle(trainImg1, center_coordinates, 10, (200,0,200), 3)
plt.imshow(trainImg1)
plt.show()
########################################################################
#reading frames from webcam and doing the proccess

cam=cv2.VideoCapture(0)
#creating objects of videoWriter for writing video
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out_avg = cv2.VideoWriter('freak avg.avi',fourcc, 6,(int(cam.get(3)),int(cam.get(4))))
out_poly = cv2.VideoWriter('freak poly.avi',fourcc, 6,(int(cam.get(3)),int(cam.get(4))))

while True:
    #reading frames
    ret, QueryImgBGR=cam.read()
    QueryImgBGR_copy = QueryImgBGR.copy()
    QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY) #turning frame to gray
    
    if ret:

        images.append(QueryImg)

    if len(images) > 2: #skiping till we have seen more than 2 frames

        #center function will detect the object in passed frame
        #and retrun center and border of detected object
        #we will perform this function in last 3 frames

        C0 = center(images[2]  , detector , freak , bf , trainDesc)
        C1 = center(images[1]  , detector , freak , bf , trainDesc)
        C2 = center(images[0]  , detector , freak , bf , trainDesc)

        if C0[0] and C1[0]: #if we have detected object in last 2 frames based on num of matches
            #drawing an arrow(green) from avg of keypoints of prevoius frame to 
            #avg of keypoints of current frame 
            cv2.arrowedLine(QueryImgBGR, C1[1], C0[1], (0,255,0), 4)
            cv2.arrowedLine(QueryImgBGR_copy, C1[3], C0[3], (0,255,0), 4)

        if C2[0] and C1[0]:
            #drawing an arrow(red) from avg of keypoints of -2 frame to 
            #avg of prevoius of current frame 
            cv2.arrowedLine(QueryImgBGR, C2[1], C1[1], (0,0,255), 4)
            cv2.arrowedLine(QueryImgBGR_copy, C2[3], C1[3], (0,0,255), 4)
            #drawing border(purple) for detected object
            cv2.polylines(QueryImgBGR,[np.int32(C2[2])],True,(255,0,255),5)
            cv2.polylines(QueryImgBGR_copy,[np.int32(C2[2])],True,(255,0,255),5)

    
        images.pop(0)
    cv2.imshow('avg',QueryImgBGR)
    cv2.imshow('poly',QueryImgBGR_copy)
    out_avg.write(QueryImgBGR)
    out_poly.write(QueryImgBGR_copy)

    if cv2.waitKey(10)==27:
        break

out_avg.release()
out_poly.release()
cam.release()
cv2.destroyAllWindows()