
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

#Spatial Features
def bin_spatial(img, size=(32,32)):
    return cv2.resize(img,size).ravel()

#Color Conversion
def convert_color(image, cspace='RGB'):
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(image)
    return feature_image
        
#Color Histogram Features
def color_hist(img, nbins=32, bins_range=(0,256)):
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features

#Hog Features
def get_hog_feature(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img,orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  visualise=vis, transform_sqrt=True, feature_vector=feature_vec)
        return features, hog_image
    else:
        features = hog(img,orientations=orient,pixels_per_cell=(pix_per_cell, pix_per_cell),
                        cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                        feature_vector=feature_vec)
        return features

#Extract Spatial, Color Histogram and Hog Features
def extract_features(imgs,cspace='RGB', spatial=True, spatial_size=(32,32),
                     hist_feature=True, nbins=32, orient=9,pix_per_cell=8, cell_per_block=2, hog_channel=0):
    features = []
    for img in imgs:
        image = mpimg.imread(img)
        img_features = []
        
        #Convert Color
        feature_image = convert_color(image, cspace=cspace)
            
        #Spatial Feature
        if spatial == True:
            img_features.append(bin_spatial(feature_image, spatial_size))
            
        #Histogram Color features
        if hist_feature == True:
            hist_features = color_hist(feature_image, nbins)
            img_features.append(hist_features)
            
        #HOG Features
        if hog_channel == 'ALL':
            hog_feature = []
            for channel in range(feature_image.shape[2]):
                hog_feature.append(get_hog_feature(feature_image[:, :, channel], orient, pix_per_cell,
                                                   cell_per_block, vis=False, feature_vec=True))
            hog_feature = np.ravel(hog_feature)
        else:
            hog_feature = get_hog_feature(feature_image[:, :, hog_channel], orient, pix_per_cell,
                                          cell_per_block, vis=False, feature_vec=True)
        img_features.append(hog_feature)
        features.append(np.concatenate(img_features))
        
    return features