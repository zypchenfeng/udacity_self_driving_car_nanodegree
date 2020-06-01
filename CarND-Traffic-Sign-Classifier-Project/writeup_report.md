## Traffic Sign Recognition
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project we built a traffic sign classifier based on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) using a deep neural network trained in TensorFlow.


The goals / steps of this project are the following:

1)     Learn how to Explore, summarise and visualise the data set

2)    Learn to Design, train and test a model architecture

3)    Use the model to make predictions on new images and analyse.

---

### 1. Examine the data set

The data set used for this project is [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) 

| Data set  |     Number of samples	        |
|:-----------------:|:-----------------------------:|
| Training        |   		34,799		  	|
| Validation      |  	    4,410         |
| Test				    |				12,630				|
| Unique classes	|				43			    	|

Every image has a dimension of **32 x 32 x 3** (width, height, channels).

Sample traffic sign image looks like

![](/home/arjun/Personal_Projects/sdc_udacity/CarND-Traffic-Sign-Classifier-Project/writeup_images/im1.png)


### 2. Data set distribution analysis

Data set distribution is not uniform, 



Train data set histogram

![](/home/arjun/Personal_Projects/sdc_udacity/CarND-Traffic-Sign-Classifier-Project/writeup_images/im2.png)

Test data set histogram

![](/home/arjun/Personal_Projects/sdc_udacity/CarND-Traffic-Sign-Classifier-Project/writeup_images/im3.png)



Valid data set histogram

![](/home/arjun/Personal_Projects/sdc_udacity/CarND-Traffic-Sign-Classifier-Project/writeup_images/im4.png)

### 3. Pre process

 Converted the images to **grayscale** and **normalize** the image as this will help the model to converge faster.


### 4. Model

My final model consisted of the following layers:

|      Layer      |                 Description                 |
| :-------------: | :-----------------------------------------: |
|      Input      |           32x32x1 grayscale image           |
|   Convolution   | 1x1 stride, valid padding, outputs 28x28x6  |
|   Activation    |                    RELU                     |
|     Pooling     | 2x2 stride, valid padding  outputs 14x14x6  |
|   Convolution   | 1x1 stride, valid padding, outputs 10x10x16 |
|   Activation    |                    RELU                     |
|     Pooling     |  2x2 stride, valid padding  outputs 5x5x16  |
|     Flatten     |                Output = 400                 |
| Fully Connected |                Output = 120                 |
|   Activation    |                    RELU                     |
| Fully Connected |                 Output = 84                 |
|   Activation    |                    RELU                     |
| Fully Connected |                 Output = 43                 |


### 5. Accuracy

![](/home/arjun/Personal_Projects/sdc_udacity/CarND-Traffic-Sign-Classifier-Project/writeup_images/im5.png)

Parameter used for the training are mentioned below

```
training_keep_prob = 0.69
rate = 0.001
EPOCHS = 40
BATCH_SIZE = 30
optimizer = AdamOptimizer
```

My final model results were:

```
accuracy for Training set: 1.000
accuracy for Validation set: 0.958
accuracy for Test set: 0.943
```



####  Test images used

![](/home/arjun/Personal_Projects/sdc_udacity/CarND-Traffic-Sign-Classifier-Project/writeup_images/im6.png)

Models were able to correct 5 images out of 6 which gives an accuracy of 83.3%



Here are the top five soft max probabilities for these images:

![](/home/arjun/Personal_Projects/sdc_udacity/CarND-Traffic-Sign-Classifier-Project/writeup_images/im7.png)



### Reflection[¶](file:///home/arjun/Downloads/report.html#Reflection)

The whole project was a great learning experience and I learned a lot about neural networks and deep learning; I have found it very interesting and bit difficult.. Looking forward to work on the next project.

There are different augmentation technique to be apply, and need to play around with more noisy data set like classifying the sign during rainfall or heavy snow.

