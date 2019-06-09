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

4. Put rectangle arround the extracted data, and also put predicted number data as a text to show the prediction.

## Result
 The accuracy of the model for the MNIST test data was 99.36% for 15 epochs.
 The automation of hadwritten number data extraction worked properly.  
 ![Screenshot from 2019-06-09 00-59-20](https://user-images.githubusercontent.com/47442084/59156077-5a0ad200-8ad0-11e9-9394-d07e5fbd61eb.png)  
 (capture of successful recognition of my handwritting)
 
 ### cf1) 
 The whole shape of captured data from my camera was (480, 640, 3)
 
 ### cf2)
 I used other windows for debugging such as window named 'predicting number of center' and 'debug_1'.  
 Window named 'predicting number of center' was used before the implemetation of automatic extracting data. It extracted data of the center squre and predicted the image. It was used to show if the model was correctly trained.  
 ![Screenshot from 2019-06-09 00-59-04](https://user-images.githubusercontent.com/47442084/59156532-fab0c000-8ad7-11e9-858d-1b12676e8bd7.png)  
 Window named 'debug_1' was used to show the every contours of the image. It was used to set the range of area of contours that will be used for the prediction.  
 ![Screenshot from 2019-06-09 16-58-29](https://user-images.githubusercontent.com/47442084/59156544-1916bb80-8ad8-11e9-8cea-ffc06cff0d14.png)

