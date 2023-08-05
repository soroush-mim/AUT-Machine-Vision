import cv2
import numpy as np

#creating an object of videoCapture for reading video
cap = cv2.VideoCapture('coins.mp4')

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
#creating an object of videoWriter for writing video
out = cv2.VideoWriter('output.avi',fourcc, 30, (int(cap.get(3)),int(cap.get(4))))
#reading video frame by frame and proccessing hough transformation
while(cap.isOpened()):
    #reading a frame from video
    ret, frame = cap.read()
    if ret: #if frame loaded seccessfully
        #turning readed fram to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #deniosing frame
        img_blur = cv2.medianBlur(gray, 13)
        # Apply hough transform on the image
        circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, img_blur.shape[0]/64, param1=275, param2=15, minRadius=30, maxRadius=35)
        # Draw detected circles
        if circles is not None:
            circles = np.uint16(np.around(circles)) #rounding circles values
            for i in circles[0, :]:
                cv2.circle(frame, (i[0], i[1]), i[2], (255, 0, 255), 3) #drawing circle on main frame

        #writing images to output video
        out.write(frame)

    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()

cv2.destroyAllWindows()
