from helper_function1 import *
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import pickle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


car_dir = "../vehicle_detection_data/vehicles/KITTI_extracted"
non_car_dir = "../vehicle_detection_data/non-vehicles/Extras"

# Concatenate feture vector
# Concatenate feature vector in a vertical vector stack
# Repeat same for test images

color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = False # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [450, 700] # Min and max in y to search in slide_window()

car_features = extract_features(car_dir, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

print("Total car images: ", len(car_features))

non_car_features = extract_features(non_car_dir, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

print("Total non-car images: ", len(non_car_features))

X = np.vstack((car_features, non_car_features)).astype(np.float64)

# Create one dimensional y vector, based on the size of training vector
car_y = np.ones(len(car_features))
non_car_y = np.zeros(len(non_car_features))
y = np.hstack((car_y, non_car_y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X_train)
# Apply the scaler to X
scaled__X_train = X_scaler.transform(X_train)
scaled__X_test = X_scaler.transform(X_test)

print("Feature vectors normalized")


print("Training feature vector dimension: ", X.shape)
print("Training y vector dimension: ",y.shape)

# Do a grid search over the dataset to find gamma and C parameter values
# save the trained classifiero
print("Starting SVM training")
#clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs), n_jobs=4, cv=10)

Cs = np.logspace(-6, -1, 10)

best_clf=None
best_score=None

for c in Cs:
    svc = LinearSVC(C=c)
    svc.fit(scaled__X_train, y_train)
    score = svc.score(scaled__X_test, y_test)
    if best_score == None or best_score < score:
        best_score = score
        best_clf = svc
    print("c: %f, score: %f" %(c,score))




pickle.dump(best_clf, open('trained_classifier.pkl', 'wb'))
pickle.dump(X_scaler, open('x_scaler.pkl', 'wb'))