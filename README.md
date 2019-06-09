# Handwritten_Numeral_Recognition_by_CNN and automatic_data_extraction
The goal of this project is to automatically extract handwritten number image data from the video and predict the number by the CNN model trained with MNIST data. cv2's contour related methods are used for the automatic data extraction.

## Requirements
・python3  
・tensorflow
  -1.13.1  
・numpy
  -1.16.2  
・openCV (cv2)
  -4.1.0  

## Process
1. Train the CNN model using MNIST example data from tensorflow module.  
In this model 3 Convolutional layers are used, and two Fully Connected layers.
The shape of input data set is (None, 28*28) and the output data is (None, 10) for recognition of 10 nimbers of 0 to 9.

2. Capture the video image in the while loop and using contour related methods in cv2 to extract the handwritten data.  
To clearly extract contours from the image, use the cv2.threshold() method to make the contours more vivid. Use the cv2.findContours() method to extract the contours.
![Screenshot from 2019-06-09 16-01-47](https://user-images.githubusercontent.com/47442084/59156073-4a8b8900-8ad0-11e9-9e4a-d3e1f2525afa.png)

