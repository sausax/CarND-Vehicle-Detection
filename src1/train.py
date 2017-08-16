from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
import pickle
import cv2
import glob
import time

from util import *


car_images = glob.glob('/home/saurabh/git_repos/CarND-Vehicle-Detection/vehicle_detection_data/vehicles/**/*.png')
noncar_images = glob.glob('/home/saurabh/git_repos/CarND-Vehicle-Detection/vehicle_detection_data/non-vehicles/**/*.png')


car_features = extract_features(car_images)
notcar_features = extract_features(noncar_images)

X = np.vstack((car_features, notcar_features)).astype(np.float64)  
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# Fit a per-column scaler
X_scaler = StandardScaler().fit(X_train)

# Apply the scaler to X
#scaled__X_train = X_scaler.transform(X_train)

#scaled__X_test = X_scaler.transform(X_test)

Cs = np.logspace(-6, -0.001, 20)
Cs = np.linspace(0.1, 2, 20)

best_clf=None
best_score=None

for c in Cs:
    svc = LinearSVC(C=c, random_state=42)
    svc.fit(X_train, y_train)
    score = svc.score(X_test, y_test)
    if best_score == None or best_score < score:
        best_score = score
        best_clf = svc
    print("c: %f, score: %f" %(c,score))

print("Best score: %f" %(best_score))

pickle.dump(best_clf, open('trained_classifier.pkl', 'wb'))
pickle.dump(X_scaler, open('x_scaler.pkl', 'wb'))

