# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/bayalievu/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 31367
* The size of the validation set is 7842 
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Exploratory visualization of the data set and bar chart showing how the data samples spread are in the ipython notebook file in cells 3 and 12.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

For preprocessing i did only normalization to using formula (pixel-255)/255 so that mean would be close to zero.

I didn't convert images to grayscale because i thought some signs with specific colors would lose its informative quality for the neural network.

I didn't generate additional data because my network were trained beyond 0.93 limit using existing data. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the same layers as original LeNet network except that input layer is 32x32x3 and output is 43:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    |1x1 stride,  valid padding 10x10x16 |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6 				|
| Flatten | Input = 5x5x16. Output = 400 |
| Fully Connected |  Input = 400. Output = 120 |
| RELU					|												|
| Fully Connected |  Input = 120. Output = 84 |
| RELU					|												|
| Fully Connected |  Input = 84. Output = 43 |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an standard AdamOptimizer with 20 Epochs and 150 Batch size. Learning rate was 0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.937 

If a well known architecture was chosen:
* What architecture was chosen? LeNet
* Why did you believe it would be relevant to the traffic sign application? First of all in udacity classroom the instructor advised so and second reason is that i believe by intuition that LeNet is capable of classifying 43 classes of traffic signs if it works ok for the english characters.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 It worked for the pictures found from the internet so i believe the model is ok.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

The five German traffic signs and the results of my classifier are shown in ipython notebook in the github repo.
In general i have 80% accuracy which is ok for dataset which is totally unknown to my trained network.
The pictures were taken from https://github.com/jeremy-shannon/CarND-Traffic-Sign-Classifier-Project so that i don't need to deal with finding and downloading the pictures.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road Work      		| Road Work   									| 
| Right-of-way at the next intersection     			| Right-of-way at the next intersection 										|
| General caution					| General caution											|
| Keep Right     		| Vehicles over 3.5 metric tons prohibited					 				|
| Speed limit 30			| Speed Limit 30      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 10th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a road work sign (probability of 0.98), and the image does contain a road work sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .98         			| Road work sign   									| 
| .007     				| Roundabout mandatory 										|
| .002					| Dangerous curve to the left										|
| .001	      			| Ahead only					 				|
| .0002				    | Turn right ahead      							|


For the second image the model is relatively sure that this is a Right-of-way at the next intersection
 (probability of 0.99). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Right-of-way at the next intersection   									| 
| .002     				|  		Beware of ice/snow								|
| .003					| 				Slippery road						|
| .001	      			| 				Double curve 				|
| .0002				    |     Priority road  							|
 

For the third image the model is relatively sure that this is a General caution (probability of 0.99). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| General caution  									| 
| .002     				|  		24								|
| .003					| 				26						|
| .001	      			| 				27 				|
| .0002				    |     31  							|
 
For the 4th image the model made wrong decision (probability of 0.95) Vehicles over 3.5 metric tons prohibited. The correct class Keep Right only had 0.003 probalility. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .95         			|   	Vehicles over 3.5 metric tons prohibited								| 
| .002     				|  		17							|
| .003					| 				19						|
| .003      			| 				Keep right 				|
| .0002				    |     20  							| 

For the 5th image the model is relatively sure that this is a Speed limit 30 (probability of 0.63). The probability is low because the Speed limit signs look like each other. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .63         			| Speed limit 30  									| 
| .15     				|  		Speed limit 50								|
| .13					| 				Speed limit 80						|
| .06	      			| 				40 				|
| .01				    |     10  							|

