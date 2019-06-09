import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import cv2
#import pprint
from MNIST_model import *
tf.reset_default_graph()
'''
print(tf.__version__)
print(np.__version__)
print(cv2.__version__)
1.13.1
1.16.2
4.1.0
'''

# cell class for predicting the number image data
net = cell()

def preprocessing1(img):
    # This preprocessing is to take the data of the center of the image
    # img data I used: img.shape = (480, 640, 3)
    img = img[190:290,270:370]
    # color -> gray ((100, 100, 3) -> (100,100,1))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Put some blur to make similar data with MNIST data
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # Change the shape of the data to MNIST data shape
    img = cv2.resize(img, (28, 28))

    #debugging using matplotlib.pyplot (you can see what kind of data your taking)
    #plt.imshow(img, cmap="Greys", interpolation="nearest")
    #plt.show()

    # Use threshold to clearly distinguish the black and white
    res, img = cv2.threshold(img, 110 , 255, cv2.THRESH_BINARY)
    img = 255 - img
    img = img.astype(np.float32)
    # scaling the data
    img /= 255
    # reshape the data to use it as an input of MNIST data model
    img = np.array(img).reshape(1,784)
    return img

def preprocessing2(img):
    # color -> gray ((x,x,3) -> (x,x,1))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (28, 28))
    #debugging using matplotlib.pyplot (you can see what kind of data your taking)
    #plt.imshow(img, cmap="Greys", interpolation="nearest")
    #plt.show()

    # Use threshold to clearly distinguish the black and white
    res, img = cv2.threshold(img, 110 , 255, cv2.THRESH_BINARY)
    img = 255 - img
    img = img.astype(np.float32)
    # scaling the data
    img /= 255
    # reshape the data to use it as an input of MNIST data model
    img = np.array(img).reshape(1,784)
    return img

# capture the data from camera
capture = cv2.VideoCapture(0)
# initialize the Window
#cv2.namedWindow("debug_1") # This window is for debugging the extracted data using contour method (showing the contour of whole display)
cv2.namedWindow("predicting number") # This window will automatically extract the data of number written in the paper and predict it
#cv2.namedWindow("predicting number of center") # This window will predict the data of the center

while True:
    # read the data from capture
    ret, frame = capture.read();

    # changing to gray scale
    img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Threshold Filtering
    retval, dst = cv2.threshold(img_gray, 110, 255, cv2.THRESH_BINARY_INV )
    # inversion of black and white
    dst = cv2.bitwise_not(dst)
    retval, dst = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    #extract the contour from the filtered data
    contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # code for debugging using contour
    #ret, dst = capture.read();
    #dst = cv2.drawContours(dst, contours, -1, (0, 0, 255, 255), 2, cv2.LINE_AA)
    #cv2.imshow('debug_1', dst)

    # read the original data
    ret, dst = capture.read();

    for contour in contours:
        area = cv2.contourArea(contour)
        # only using contours that have area over than 400 and less than 20000
        if area<400 and 20000>area:
            continue

        x,y,w,h = cv2.boundingRect(contour)
        a = max(w,h)
        # img.shape = (480, 640, 3)
        if (a<10 or x+w/2-a*1.3/2<1 or x+w/2+a*1.3/2>479 or y+h/2-a*1.3/2 <1 or y+h/2+a*1.3/2 >639):
            continue
        # extract range is decided to make the number data to be similar with MNIST data(train data)
        img = dst[int(y+h/2-a*1.3/2):int(y+h/2+a*1.3/2),int(x+w/2-a*1.3/2):int(x+w/2+a*1.3/2),:]
        img = preprocessing2(img) # img shape => (28,28)
        # drawing rectangle around the data that is predicted
        dst = cv2.rectangle(dst,(int(x+w/2-a*1.3/2),int(y+h/2-a*1.3/2)),(int(x+w/2+a*1.3/2),int(y+h/2+a*1.3/2)),(0,255,0),2)
        # predict the data
        pred = net.predict(img)[0]
        text = "number{}".format(str(pred))
        font = cv2.FONT_HERSHEY_PLAIN
        font_size = 1
        # put the text including the prediction information
        cv2.putText(dst,text,(int(x+w/2-a*1.3/2),int(y+h/2+a*1.3/2 +16)),font, font_size,(255,255,0))

        c = cv2.waitKey(1)

    cv2.imshow('predicting number', dst)
    # get the center image (100,100,3), this is for easy checking
    #cv2.rectangle(frame,(270,190),(370,290),(0,0,255),3)
    #img = preprocessing1(frame)
    #pred = net.predict(img)[0]
    #text = "number{}".format(str(pred))
    #font = cv2.FONT_HERSHEY_PLAIN
    #font_size = 1
    #cv2.putText(frame,text,(370,290),font, font_size,(255,255,0))

    #cv2.imshow("predicting number of center", frame)
    c = cv2.waitKey(2)

    if c == 27: # Esc key
        print("Section ended")
        break

capture.release()
cv2.destroyAllWindows()
