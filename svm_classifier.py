
from feature_functions import *
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import glob
import pickle
import os
# %matplotlib inline

car1 = glob.glob('vehicles/GTI_Far/*.png')
car2 = glob.glob('vehicles/GTI_Left/*png')
car3 = glob.glob('vehicles/GTI_MiddleClose/*.png')
car4 = glob.glob('vehicles/GTI_Right/*.png')
car5 = glob.glob('vehicles/KITTI_extracted/*.png')
cars = np.concatenate((car1,car2,car3,car4,car5))

non_car1 = glob.glob('non-vehicles/Extras/*.png')
non_car2 = glob.glob('non-vehicles/GTI/*.png')
non_cars = np.concatenate((non_car1,non_car2))

#Define parameters
colorspace = 'YCrCb'
spatial_size = (32,32)
nbins = 32
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'

#Extract spatial, color histogram and hog features
def main():
	car_features = extract_features(cars, cspace=colorspace, spatial_size=spatial_size,
	                                nbins=nbins, orient=orient, pix_per_cell=pix_per_cell,
	                                cell_per_block=cell_per_block, hog_channel=hog_channel)

	noncar_features = extract_features(non_cars, cspace=colorspace, spatial_size=spatial_size,
	                                nbins=nbins, orient=orient, pix_per_cell=pix_per_cell,
	                                cell_per_block=cell_per_block, hog_channel=hog_channel)

	#Training Inputs
	X = np.vstack((car_features, noncar_features)).astype(np.float64)
	X_scaler = StandardScaler().fit(X)
	scaled_X = X_scaler.transform(X)
	y_train = np.concatenate((np.ones(len(cars)), np.zeros(len(non_cars))))

	#Split training and test sets
	X_train, y_train = shuffle(scaled_X, y_train)
	X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.15, random_state = 42)

	#Training
	svc = LinearSVC()
	svc.fit(X_train, y_train)
	accuracy = svc.score(X_test, y_test)

	print ('Accuracy: ', accuracy)
	###Pickle Classifier###
	pickle_file = 'SVMvehicles.p'
	if not os.path.isfile(pickle_file):
	    print('Saving data to pickle file...')
	    try:
	        with open('SVMvehicles.p', 'wb') as pfile:
	            pickle.dump(
	            {
	                'svm_model': svc,
	                'scaler' : X_scaler,
	                'color_space': colorspace,
	                'spatial_size': spatial_size,
	                'bins_number': nbins,
	                'orient': orient,
	                'pix_per_cell': pix_per_cell,
	                'cell_per_block': cell_per_block,
	                'hog_channel': hog_channel
	            },
	            pfile, pickle.HIGHEST_PROTOCOL)
	    except Exception as e:
	        print('Unable to save data to', pickle_file, ':', e)
	        raise
	        
	print('Data cached in pickle file.')

if __name__ == '__main__':
	main()