import cv2
import numpy as np

def HE(img , color_space , inv_color_space , name ):
    
    # convert from RGB to color_space
    img = cv2.cvtColor(img,color_space)
    
    # equalize the histogram of the luminance channel
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])

    # convert back to RGB 
    equalized_img = cv2.cvtColor(img, inv_color_space)

    cv2.imshow('equalized_img', equalized_img)
    cv2.imshow('equalized_luminance_channel', equalized_img[:,:,0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(name + '.png' , equalized_img)


def AHE(img , color_space , inv_color_space , name):

    # convert from RGB to color_space
    img = cv2.cvtColor(img,color_space)

        
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    # equalize the histogram of the luminance channel
    img[:, :, 0] = clahe.apply(img[:, :, 0])

    # convert back to RGB 
    equalized_img = cv2.cvtColor(img,inv_color_space)

    cv2.imshow('equalized_img', equalized_img)
    cv2.imshow('equalized_luminance_channel', equalized_img[:,:,0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(name + '.png', equalized_img)

indoor = cv2.imread('indoor.png')
outdoor = cv2.imread('outdoor.png')

HE(indoor , cv2.COLOR_BGR2LAB ,cv2.COLOR_LAB2BGR, 'indoorHELAB')
HE(outdoor , cv2.COLOR_BGR2LAB ,cv2.COLOR_LAB2BGR, 'outdoorHELAB')
HE(indoor , cv2.COLOR_BGR2YCrCb ,cv2.COLOR_YCrCb2BGR, 'indoorHEYCrCb')
HE(outdoor , cv2.COLOR_BGR2YCrCb ,cv2.COLOR_YCrCb2BGR, 'outdoorHEYCrCb')

AHE(indoor , cv2.COLOR_BGR2LAB ,cv2.COLOR_LAB2BGR, 'indoorAHELAB')
AHE(outdoor , cv2.COLOR_BGR2LAB ,cv2.COLOR_LAB2BGR, 'outdoorAHELAB')
AHE(indoor , cv2.COLOR_BGR2YCrCb ,cv2.COLOR_YCrCb2BGR, 'indoorAHEYCrCb')
AHE(outdoor , cv2.COLOR_BGR2YCrCb ,cv2.COLOR_YCrCb2BGR, 'outdoorAHEYCrCb')