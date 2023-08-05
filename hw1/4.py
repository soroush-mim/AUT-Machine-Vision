import cv2
import numpy as np

cap = cv2.VideoCapture(0)
n = 10 #num of frames for averaging
images = [] #list to save frames

while(True):
    ret, frame = cap.read()
	
    if ret: #if frame has fecthed correctly add it to list of frames 
        images.append(frame/n)
        
    if len(images) > n: #if number of fecthed frames is bigger than n do averaging
        result = np.sum(images[-n:], axis = 0).astype(np.uint8)
        #showing result
        cv2.imshow('result', result)
    #showing main frames
    cv2.imshow('video', frame)
    if cv2.waitKey(30) == 27: #waiting 30 miliseconds
        # ESC pressed
        print("Escape hit, closing...")
        break 
#releasing cam
cap.release()
cv2.destroyAllWindows()
