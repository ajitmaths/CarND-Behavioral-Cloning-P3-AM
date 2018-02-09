# **Behavioral Cloning**

#### The goal of this project are the following:
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Use the simulator to collect data of good driving behavior
* Test that the model successfully drives around track  without leaving the road

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

[//]: # (Image References)
[image1]: ./examples/Original_Image_Center_Left_Right.png "Original Image"
[image2]: ./examples/Cropping_and_Blurring.png "Cropping_and_Blurring"
[image3]: ./examples/Horizontal_Flip.png "Horizontal_Flip"
[image4]: ./examples/Random_Shadow.png "Random_Shadow"
[image5]: ./examples/Brightness_Shift.png "Brightness_Shift"
[image6]: ./examples/Data_Analysis.png "Data Analysis"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[image8]: ./examples/recovery_from_side.jpg "Recovery from side"
[image9]: ./startingpoint.mp4 "Starting point video - trivial model"
[image10]: ./finalvideo.mp4 "Final Tuned video"
[image11]: ./examples/model_architecture.png "Model Architecture"
[image12]: ./examples/modelsummary.png "Model Summary"
[image13]: ./examples/mean_squared_error_train_val.png "Mean Squared Error - Train and Val"

### Exploratory visualization of the dataset

I used the pandas library to calculate summary statistics of the driving_log.csv signs data set.

![alt text][image6]

![alt text][image1]

We have a biased data set with lot of samples with steering angle 0 given that the car is driving straight most of the time. This basically requires data augmentation. There is also left turn bias in the dataset.

### Data Augmentation
Following approaches has been taken -
##### 1. Flipping images and taking the opposite sign of the steering measurement

``image_flipped = np.fliplr(image)
measurement_flipped = -measurement``

##### 2. Steering Angle Corrections
From the perspective of the left camera, the steering angle would be less than the steering angle from the center camera. From the right camera's perspective, the steering angle would be larger than the angle from the center camera. So there is a `correction` factor that need be applied to right and left steering angles. Figuring out how much to add or subtract from the center angle requires some experimentations. Value chosen was `0.25`.
During training, you want to feed the left and right camera images to your model as if they were coming from the center camera. This way, you can teach your model how to steer if the car drifts off to the left or the right. Below is the example of recovery from side.
![alt text][image8]

##### 3. Driving Counter-Clockwise
First set of data that i collected only used the first track in a clock-wise direction and hence the data is biased towards left turns. In order to combat the left turn bias I took the car around and recorded counter-clockwise laps around the track. Alos, driving counter-clockwise is also like giving the model a new track to learn from, so the model will generalize better.


### Step to Reduce Overfitting

##### 1. Dropout Layers
By using the dropout layer (in commai model) I reduced the overfitting to help the model generalize better. Performing dropout for a layer will randomly dropout or deactivate a set number of nodes for the layer dropout is performed to try to reduce overfitting.

##### 2. Training and Validation
The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

##### 3. `fit_generator`

One great advantage about fit_generator() besides saving memory is user can integrate random augmentation inside the generator, so it will always provide model with new data to train on the fly. This helps the model generalize well. I tried both the `fit_generator` and ``fit``.

``model.fit_generator(train_generator, steps_per_epoch= samples_per_epoch, validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose = 1)``

With the generator above, I defined batch_size = 32 , that means it will randomly taking out 32 samples from features and labels to feed into each epoch until an epoch hits its sample limit defined by `samples_per_epoch`. Then fit_generator() destroys the used data and move on repeating the same process in new epoch.

##### 4. Batch Normalization and Dropouts
Batch normalization is a technique that is typically done by transforming the mini batch data to zero mean and unit variance before the non-linearity function to try to help improve the gradients of the outputs of the non-linear functions.
Both dropouts and batch normazliations help resolve the vanishing the gradient descent problem.

### Model Parameter Tuning
The model used an adam optimizer, so the learning rate was not tuned manually. The ``AdamOptimizer`` uses Adam algorithm to control the learning rate. Adam offers several advantages over the simple ``GradientDescentOptimizer``.  Stochastic gradient descent which maintains a single learning rate for all weight updates and the learning rate does not change during training. Foremost is that it uses moving averages of the parameters (momentum). Simply put, this enables Adam to use a larger effective step size, and the algorithm will converge to this step size without fine tuning. The main down side of the algorithm is that Adam requires more computation to be performed for each parameter in each training step (to maintain the moving averages and variance, and calculate the scaled gradient); and more state to be retained for each parameter (approximately tripling the size of the model to store the average and variance for each parameter).

### Training Data - Creation

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I made use of the left and right camera angles also, where I modified the steering angles by 0.25. This helped up the number of test cases, and these help cases where the car is off center and teaches it to steer back to the middle.I used 3 epochs for enough training. In the beginning I picked three images from the .csv file, one with negative steering, one with straight, and one with right steering. Trained a model with just those three images and see if you can get it to predict them correctly. After that i started adding more training data

### Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I followed following guidelines for data collection:

* one lap of center lane driving
* one lap of recovery driving from the sides
* one lap focusing on driving smoothly around curves

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover to the center of the road. These images show what a recovery looks like.

![alt text][image8]
To augment the data sat, I also flipped images and angles thinking that this would ... For example.

After the collection process, I had 26772 number of data points. I then preprocessed this data which consisted of the following pipelines: Gaussian Blur, Horizontal Flip and Brightness Shift. Here are the results starting the Original Image through the end of the pipeline.


| __Gaussian Blur__     |     __Horizontal Flip__     |
|:---------------------:|:-------------------------:|
|![alt text][image1]    |![alt text][image2]        |


| __Brightness Shift__  |    __Random Noise__       |
|:---------------------:|:-------------------------:|
|![alt text][image3]    |![alt text][image5]        |


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. Since this model outputs a single continuous numeric value, appropriate error metric is mean squared error. If the mean squared error is high on both a training and validation set, the model is underfitting. If the mean squared error is low on a training set but high on a validation set, the model is overfitting. The ideal number of epochs was 3 as evidenced by convergence of mean squared error to zero for both training and validation data set. I used an adam optimizer so that manually training the learning rate wasn't necessary.

In Keras, ``model.fit_generator()`` methods have a verbose parameter that tells Keras to output loss metrics as the model trains.
Setting model.fit(verbose = 1) will
* output a progress bar in the terminal as the model trains.
* output the loss metric on the training set as the model trains.
* output the loss on the training and validation sets after each epoch.

### Model Architecture and Reasoning
I experimented with two models. One is the Nvidia model and the other one is comm.ai model. It takes approximately 380/step to train these models (EC2 instance g2.8xLarge) - Batch size 32, Sample size ~20K. So it takes ~4 Hours to train the complete model.

Final Model consists of the following:

![alt text][image12]

The model includes RELU layers to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer.

#### Solution Design Approach

The overall strategy for deriving a model architecture was to figure out the simple model which trains fast yet be able to reproduce the optimal result.
I started out with the one lap of training data recording. My first step was to use a convolution neural network model with trivial model. Trivial Model consisted of a input pipeline along with Lambda, Cropping and Conv2D followed by `model.fit` using the Adam optimizer. Clearly, as you can see in the video
__./startingpoint.mp4 "Starting point video - trivial model"__ Car has the difficulty staying on the road.

In order to gauge how well the model was working, I split the image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model to increased the validation and

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track and these cases i used additional data to improve the driving behavior in these cases. I used both __commai__ and __Nvidia__ model - so it gave a very good start.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. The final video is __./finalvideo.mp4 "Final video - Nvidia model"

### Final Model Architecture

The final model architecture uses the Nvidia model and consisted of a convolution neural network with the following layers and layer sizes explained in the __Model Architecture and Reasoning section__.


Below is the Mean Square error for Validation and Training data set.

![alt text][image13]


### Submission code

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


