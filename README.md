# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
The scope of this project is to detect moving vehicle in a video. This is the 5th project in Term 1 of Self Driving Car nanodegree from Udacity.
### Installation & Resources
1. Anaconda Python 3.5
2. Udacity [Carnd-term1 starter kit](https://github.com/udacity/CarND-Term1-Starter-Kit) with miniconda installation 
3. Udacity [provided data](https://github.com/udacity/CarND-Vehicle-Detection)
4. [Vehicle Datasets](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [Non Vehicle Datasets](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)

### Files
`training.py` : contain classifier of vehicle and non-vehicle training
`extract_features.py`: contains help code to extract features
`detect_vehicles.py`: contains pipeline to detect vehicle in video

### Goals
1. Apply Histogram of Oriented Gradients to extract object features for classification training.
2. Explain a method that trains a classifier with extracted features.
3. Apply sliding window technique on video frame to search for vehicle. Explain scaling and window overlap.
4. Show image examples from pipeline testing. How to optimize the classifier.
5. Provide final output video with minimal false positive.
6. Describe how to filter false positives.
7. Discussion

### I. Histogram of Oriented Gradients (HOG)
One of the common feature discriptor being used to extract object features is HOG which is to find the direction of gradient in pixels and combine them into histogram bins. The advantage of HOG is we can define unique direction signature of

The function define HOG is from `line#` to `line#` in `extract_features.py` file. To extract HOG feature, I use the `hog` fucntion  from `skimage.feature` library. 

Some variables need to be provided for the function is `image`, `orientations` of HOG, `# of pixels of each cell`, `# of cells in each block`. After some experiments, I settled with the variables in `line#` to `line#`. Here are some examples of extracting HOG features.
###IMAGE###
### II. Train classifier with extracted features
To prepare data to train a classifier, I combined some features in each images such as `spatial bin`, `color histogram` and `hog` into an array. The choice of training method is Support Vector Machine (SVM), a series of features from `vehilces` and `non-vehicles` datasets were fed into the SVM model with label `0 for car` and `1 for noncar`. `Line#` to `line#` in `training.py` code.

### III. Sliding Window Search
To extract information from video frame to feed into classifier, I apply sliding windows with size `64x64` which is the same size as training images. The rate of sliding is `cell_per_step` with defined `pixels_per_cell`.
The steps in either `x` or `y` direction can be broken down as following steps:
`cells_per_direction = total_pixels_length//pixels_per_cell`
`cells_per_window = window_side//pixels_per_cell`

`steps_per_window = cells_per_window//cells_per_step` 
or `step_per_window = window_side//(pixels_per_cell*cells_per_step)`

`steps = [(cells_per_direction/cells_per_window)-1]*steps_per_window`
or `steps = [(total_pixels_length/window_size)-1]*steps_per_window`

For example: a `128 pixels` width with `64x64 pixels` window that slides at the rate of `2 cells per step` with `8 pixels per cell`. `steps_per_window = 64//(8*2) = 4`, `steps = ((128/64)-1)*4 + 1= 5 

