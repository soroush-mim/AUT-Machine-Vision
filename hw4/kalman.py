import cv2
import numpy as np
import matplotlib.pyplot as plt

########################################################################
#kalman class that we will use for kalman filter
class KalmanFilter:
    #initializing kalman filter and its parameters
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)


    def predict(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        x, y = int(predicted[0]), int(predicted[1])
        return x, y

########################################################################

def center(QueryImg  , detector , flann , trainDesc , predict = None):
    """this function calculate sift keypoints and discriptors
        for an image then matches these features with keypoints
        of train img and detect the object and return center and border
        of object,  it also predicts the next position of object
        based on the current positon and a kalman filter, i should note that 
        kalman filter uses all previous points to predict

    Args:
        QueryImg ([2d np array]): [input img]
        detector ([CV2 detector]): [detctor that we want to use  (here we use SIFT)]
        flann ([cv2.FlannBasedMatcher]): [matcher for mathcing features]
        trainDesc ([cv2 descriptor]): [descriptors of train img]
        predict ([none or a 2*1 np array]): position of last predicted point by kalman

    Returns:
        return 4 vars
        first one tell us that we have detected object in img
        seconde one is avg point of mathced keypoints
        third one is predicted position
        fourth one is the border of detcted object
    """

    #computing keypoints and descriptors for input img
    queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)

    #matching calculated descriptors and train img descriptors
    matches=flann.knnMatch(queryDesc,trainDesc,k=2)

    # a list for keeping only good matches
    goodMatch=[]
    
    # The distance ratio between the two nearest matches of a considered
    # keypoint is computed and it is a good match when this value is below
    # a threshold
    for m,n in matches:
        if(m.distance<0.75*n.distance):
            goodMatch.append(m)
    
    #if we have matched enough keypoints for detection
    if(len(goodMatch)>MIN_MATCH_COUNT):

        goodMatch = sorted(goodMatch , key= lambda x: x.distance)

        tp=[] # keeping mathced points in train image
        qp=[] # keeping mathced points in input image

        for m in goodMatch[:int(len(goodMatch)*.8)]:
            tp.append(trainKP[m.trainIdx].pt)
            qp.append(queryKP[m.queryIdx].pt)

        tp,qp=np.float32((tp,qp))

        #finding borders of detected object
        H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
        h,w=trainImg.shape
        trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
        queryBorder=cv2.perspectiveTransform(trainBorder,H)
        
        qp = queryBorder[0]

        #avg of vertixes of detected border
        detected_center = np.int64(np.sum(queryBorder[0] , axis=0) / queryBorder[0].shape[0])
        print("Match Found")
        
        #predicting next position with kalman based on current position
        predicted = kf.predict(detected_center[0] , detected_center[1])

        return True ,detected_center ,predicted , queryBorder
        
    else:

        print("Not Enough match found-")
        print(len(goodMatch),MIN_MATCH_COUNT)

        if predict:

            #predicting next position with kalman based on prevoius prediction
            predict = kf.predict(predict[0] , predict[1])
        return False , predict

########################################################################
#initialize parameters
#minimum match points to accept detection
MIN_MATCH_COUNT=20

#creating kalman object
kf = KalmanFilter()

#creating SIFT object
detector=cv2.xfeatures2d.SIFT_create()

#initializing matcher, we will use histogram based metric and 
# 2 nearest neighbor method
FLANN_INDEX_KDITREE=0
flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
flann=cv2.FlannBasedMatcher(flannParam,{})

#this var will keep last predicted position by kalman filter
predict = None
########################################################################
#computing sift keypoints and descriptors of train img and plotting them
#i also plotted the avg point of keypoints as a purple circle
#reading train img
trainImg=cv2.imread("train.jpg",0)

#computing sift keypoints and descriptors
trainKP,trainDesc=detector.detectAndCompute(trainImg,None)

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
out_avg = cv2.VideoWriter('KalmanFilter.avi',fourcc, 12,(int(cam.get(3)),int(cam.get(4))))

while True:

    #reading frames
    ret, QueryImgBGR=cam.read()
    QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY) #turning frame to gray

    if ret:

        #center function will detect the object in passed frame
        #and retrun center and border of detected object it also 
        #will return predicted position for object in next frame
        C = center(QueryImg  , detector , flann , trainDesc , predict)

        if C[0]: #if we have detected object

            predict = C[2] #predicted position
            #drawing border of detected object(purple)
            cv2.polylines(QueryImgBGR,[np.int32(C[3])],True,(255,0,255),5)
            #drawing an arrow from center of object to the predicted positon
            #of center for next frame(green)
            cv2.arrowedLine(QueryImgBGR, C[1], C[2], (0,255,0), 4)
            #drawing a circle in the center of detected object(blue)
            cv2.circle(QueryImgBGR,  C[1], 10, (255,0,0), 3)
            #drawing a circle in the predicted position fo center for the detected object(red)
            cv2.circle(QueryImgBGR,  C[2], 10, (0,0,255), 3)

        else:
            if C[1]: #if we havent detect object

                #drawing a circle in the predicted position fo center for the detected object(red)
                #based on last prediction
                cv2.circle(QueryImgBGR,  C[1], 10, (0,0,255), 3)
                predict =  C[1]


    cv2.imshow('result',QueryImgBGR)
    out_avg.write(QueryImgBGR)

    if cv2.waitKey(25)==27:
        break

out_avg.release()
cam.release()
cv2.destroyAllWindows()