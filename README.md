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
When using extracted contours, make sure not to use contours that are too small or too big.
![Screenshot from 2019-06-09 16-10-52](https://user-images.githubusercontent.com/47442084/59156129-5297f880-8ad1-11e9-9db7-93ba0f25ddb3.png)


3. For every extracted contours, find the bounding rectangles and extract the bounding square image data and preprocess the extracted square data to use as an input to the MNIST_CNN_predict model.  
when extracting the square image data, be careful not to extract the just size of the contour. Instead, include some white black in the surrounding of the image to make the form of the data similar with MNIST data that had trained the model
![Screenshot from 2019-06-09 16-26-23](https://user-images.githubusercontent.com/47442084/59156285-6a707c00-8ad3-11e9-9b4f-3e632110b94a.png)

4. 

