# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
The scope of this project is to detect moving vehicle in a video. This is the 5th project in Term 1 of Self Driving Car nanodegree from Udacity.
### Installation & Resources
1. Anaconda Python 3.5
2. Udacity [Carnd-term1 starter kit](https://github.com/udacity/CarND-Term1-Starter-Kit) with miniconda installation 
3. Udacity [provided data](https://github.com/udacity/CarND-Vehicle-Detection)
4. [Vehicle Datasets](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [Non Vehicle Datasets](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)

### Files
`vehicle_detection.py` : contain classifier of vehicle and non-vehicle training

`feature_functions.py`: contains help code to extract features

`svm_classifier.py`: contains pipeline to detect vehicle in video

### Goals
1. Apply Histogram of Oriented Gradients to extract object features for classification training.
2. Explain a method that trains a classifier with extracted features.
3. Apply sliding window technique on video frame to search for vehicle. Explain scaling and window overlap.
4. Show image examples from pipeline testing. How to optimize the classifier.
5. Provide final output video with minimal false positive. Describe how to filter false positives.
6. Discussion

### I. Histogram of Oriented Gradients (HOG)
(`Line 39` to `Line 50` in `feature_functions.py`)
One of the common feature discriptor being used to extract object features is HOG which is to find the direction of gradient in pixels and combine them into histogram bins. The advantage of HOG is we can define unique direction signature of certain object. To extract HOG feature, I use the `hog` fucntion  from `skimage.feature` library. 

Some variables need to be provided for the function is `image`, `orientations` of HOG, `# of pixels of each cell`, `# of cells in each block`. After some experiments, I settled with the variables in `line 22` to `line 28` in `svm_classifier.py`. Here are some examples of extracting HOG features.
###IMAGE###
### II. Train classifier with extracted features
`Line 31` to `Line 55` in `svm_classifier.py`
To prepare data to train a classifier, I combined some features in each images such as `spatial bin`, `color histogram` and `hog` into an array. The choice of training method is Support Vector Machine (SVM), a series of features from `vehilces` and `non-vehicles` datasets were fed into the SVM model with label `0 for car` and `1 for noncar`.
The choice of parameters for training is after several implementation, the current choice has relatively good speed to train with high accuracy.

### III. Sliding Window Search
(`Line 34` to `Line 105` in `vehicle_detection.py`)
To extract information from video frame to feed into classifier, I apply sliding windows with size `64x64` which is the same size as training images. The rate of sliding is `cell_per_step` with defined `pixels_per_cell`.
The steps in either `x` or `y` direction can be broken down as following steps:

`cells_per_direction = total_pixels_length//pixels_per_cell`

`cells_per_window = window_side//pixels_per_cell`

`steps_per_window = cells_per_window//cells_per_step` 

or `step_per_window = window_side//(pixels_per_cell*cells_per_step)`

`steps = [(cells_per_direction/cells_per_window)-1]*steps_per_window`

or `steps = [(total_pixels_length/window_size)-1]*steps_per_window`

For example: a `128 pixels` width with `64x64 pixels` window that slides at the rate of `2 cells per step` with `8 pixels per cell`. `steps_per_window = 64//(8*2) = 4`, `steps = ((128/64)-1)*4 + 1= 5`
#### Scaling
Combine a different set of scaling in each frame helps to detect `car object` even they are larger or bigger as they moving far away on the road. Higher scale decreases the size of the frame image, and lower scale increases the size.
#### Region of Interest
To improve the efficiency of feature extraction with less false positive in detection, window sliding is purposedly only applied in the region of interest. Region of interest in this case is the lower half of the image which contains the road and moving vehicles.


### IV. Test Image Examples
With extracted features from each window slide, the SVM classifer can be used to detect if the feature is a vehicle or not. If SVM confirms the current window is vehicle, we can rescale the coordinate of the window back to its original scale and save these location coordinates. Below is a test image showing the detected vehicle whose coordinates are drawn in blue boxes. Many other examples are in `test_images` folder. 
![test1](https://cloud.githubusercontent.com/assets/23693651/23692815/ed36874c-039e-11e7-9659-3b9b892d3215.png)

#### Heatmap
(`Line 108` to `Line 116` in `vehicle_detection.py`)
Assume a normal frame has no heat with black background of all `0` values. Everytime a vehicle is detected, we can add heat (or a value of `1`) to that detected region. When many detected regions overlap each other at the same location, we have a high heat region around that location. By using `label` from `scipy.ndimage.measurements`, we can define how many high heated regions and one average coordinate of that region.  
To further filter out the false positive, a threshold can be applied so that the heatmap only contains more stable detection which is a high heat region.

![heatmap](https://cloud.githubusercontent.com/assets/23693651/23692817/f0ef362c-039e-11e7-8d8b-4a6178c07e69.png)

### V. Output Video
[Output Video](https://)
Every frame in the video is processed as described in Sliding Window Search. 
To efficiency set a threshold to remove false positive in `heatmap`. To smooth the detection in each frame, an average of detected vehicles coordinates in `10` current frame is used label the vehicle.

### VI. Discussion
False positives are still found in the video. I think some factors may improve this is to do more tuning on the parameters such as heatmap threshold, scales and number of average frames. In the video, black car seems to get detected easier than white car, more trial in training in different colorspace can also be considered to see if that helps to better detecting bright color vehicles.

