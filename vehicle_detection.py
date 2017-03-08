from feature_functions import *
import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from scipy.ndimage.measurements import label
from collections import deque
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from moviepy.editor import VideoFileClip

#Load Training Pickle
pickle_file = 'SVMvehicles.p'
with open(pickle_file, 'rb') as f:
    pickle_data = pickle.load(f)
    svc = pickle_data['svm_model']
    X_scaler = pickle_data['scaler']
    colorspace = pickle_data['color_space']
    spatial_size = pickle_data['spatial_size']
    nbins = pickle_data['bins_number']
    orient = pickle_data['orient']
    pix_per_cell = pickle_data['pix_per_cell']
    cell_per_block = pickle_data['cell_per_block']
    hog_channel = pickle_data['hog_channel']
    del pickle_data
    
print('Data and modules loaded')

def find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                spatial_size, hist_bins):
    
    draw_image = np.copy(image)
    scaled_image = image.astype(np.float32)/255
    
    #Region of Interest (ROI)
    image_roi = scaled_image[ystart:ystop,:,:]
    search_image = convert_color(image_roi,cspace='YCrCb')
    
    #Scale ROI
    imshape = search_image.shape
    if scale != 1:
        search_image = cv2.resize(search_image, (np.int(imshape[1]/scale), (np.int(imshape[0]/scale))))

    #Blocks of the entire region of interest
    nx_blocks = (search_image.shape[1]//pix_per_cell) - cell_per_block + 1
    ny_blocks = (search_image.shape[0]//pix_per_cell) - cell_per_block + 1
    
    #Window without steps
    window = 64
    window_blocks = (window//pix_per_cell) - cell_per_block + 1
    cell_per_window = window//pix_per_cell
    x_windows = search_image.shape[1]//window
    y_windows = search_image.shape[0]//window
    
    #With steps
    cell_per_step = 2
    step_per_window = cell_per_window//cell_per_step
    nxsteps = (x_windows-1)*step_per_window
    nysteps = (y_windows-1)*step_per_window
    
    #Extract hog features without feature vector in all channel
    hog1 = get_hog_feature(search_image[:,:,0], orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_feature(search_image[:,:,1], orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_feature(search_image[:,:,2], orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    car_boxes = []
    for xstep in range(nxsteps):
        for ystep in range(nysteps):
            #Cell position
            ypos = ystep*cell_per_step
            xpos = xstep*cell_per_step
            #Extract HOG for each step
            hog_feat1 = hog1[ypos:ypos+window_blocks, xpos:xpos+window_blocks].ravel()
            hog_feat2 = hog2[ypos:ypos+window_blocks, xpos:xpos+window_blocks].ravel()
            hog_feat3 = hog3[ypos:ypos+window_blocks, xpos:xpos+window_blocks].ravel()
            hog_features = np.hstack((hog_feat1,hog_feat2,hog_feat3))
            
            #Pixel position
            x_left = xpos*pix_per_cell
            y_top = ypos*pix_per_cell
            
            #Window Image
            sub_img = cv2.resize(search_image[y_top:y_top+window, x_left:x_left+window],(64,64))
            
            #Spatial and Color Features
            spatial_features = bin_spatial(sub_img, spatial_size)
            color_hist_features = color_hist(sub_img, nbins=hist_bins, bins_range=(0,256))
            
            #Combine test features
            test_features = X_scaler.transform(np.concatenate((spatial_features,color_hist_features,hog_features)).reshape(1,-1))
            
            prediction = svc.predict(test_features)
            if prediction == 1:
                xleft_draw = np.int(x_left*scale)
                ytop_draw = np.int(y_top*scale) + ystart
                win_draw = np.int(window*scale)
#                 cv2.rectangle(draw_image,(xleft_draw,ytop_draw), (xleft_draw+win_draw, ytop_draw+win_draw),(0,0,255),6)
                car_boxes.append(((xleft_draw,ytop_draw),(xleft_draw+win_draw, ytop_draw+win_draw)))
    
    return car_boxes

#Add head
def add_heat(heatmap, car_boxes):
    for box in car_boxes:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap

#Heatmap threshold
def heat_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap

#Draw car
def draw_car_box(image, labels):
    for car_number in range(1,labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        car_y = np.array(nonzero[0])
        car_x = np.array(nonzero[1])
        top_left = (np.min(car_x),np.min(car_y))
        bottom_right = (np.max(car_x),np.max(car_y))
        cv2.rectangle(image, top_left, bottom_right, (0,0,255), 6)       
    return image

#Pipeline detect vehicle in image
n = 15 #Number of current frames
car_topLeft = [deque(maxlen=n) for x in range(10)]
car_lowRight = [deque(maxlen=n)for x in range(10)]
def pipeline(image):
	ystart = 400
	ystop = 650
	scales = [0.75,1,1.25,1.5]
	car_boxes = []
	for scale in scales:
		car_boxes.extend(find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins = nbins))
	heat = np.zeros_like(image[:,:,0]).astype(np.float)
	heat = add_heat(heat, car_boxes)
	heat = heat_threshold(heat,3)
	heatmap = np.clip(heat,0,255)
	labels = label(heat)
	draw_img = np.copy(image)

	for car_number in range(1,labels[1]+1):
		nonzero = (labels[0] == car_number).nonzero()
		car_y = np.array(nonzero[0])
		car_x = np.array(nonzero[1])
        #Find current top left and low right coordinates of current vehicle
		topLeft_current = (np.min(car_x),np.min(car_y))
		lowRight_current = (np.max(car_x),np.max(car_y))
        #Add current coordinates into list of current frames
		car_topLeft[car_number].append(topLeft_current)
		car_lowRight[car_number].append(lowRight_current)
		box_left = list(car_topLeft[car_number])
		box_right = list(car_lowRight[car_number])
		m = len(box_left)
        #Average coordinates of current n frames
		top_left = (sum (k[0] for k in box_left)//m, sum(k[1] for k in box_left)//m)
        
		low_right = (sum (k[0] for k in box_right)//m, 
                           sum(k[1] for k in box_right)//m)
		if m > 5:
			cv2.rectangle(draw_img, top_left, low_right, (255,255,0), 5)
	return draw_img

def main():
	test_output = 'project_vid1.mp4'
	clip1 = VideoFileClip('project_video.mp4')
	test_clip = clip1.fl_image(pipeline)
	test_clip.write_videofile(test_output, audio=False)

	print ('Done processing video')

# def main():
# 	test_output = 'project_clip.mp4'
# 	clip1 = VideoFileClip('project_video.mp4').subclip(4,14)
# 	test_clip = clip1.fl_image(pipeline)
# 	test_clip.write_videofile(test_output, audio=False)

# 	print ('Done processing video')

if __name__ == '__main__':
	main()