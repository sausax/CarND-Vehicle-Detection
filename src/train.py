from helper_function1 import *
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import pickle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA



car_dirs = ["../vehicle_detection_data/vehicles/KITTI_extracted",
            "../vehicle_detection_data/vehicles/GTI_Far",
            "../vehicle_detection_data/vehicles/GTI_Left",
            "../vehicle_detection_data/vehicles/GTI_Right",
            "../vehicle_detection_data/vehicles/GTI_MiddleClose"]
non_car_dirs = ["../vehicle_detection_data/non-vehicles/Extras",
                "../vehicle_detection_data/non-vehicles/GTI"]

# Concatenate feture vector
# Concatenate feature vector in a vertical vector stack
# Repeat same for test images

features = get_features()

car_features = extract_features(car_dirs)

print("Total car images: ", len(car_features))

non_car_features = extract_features(non_car_dirs)

print("Total non-car images: ", len(non_car_features))

X = np.vstack((car_features, non_car_features)).astype(np.float64)

# Create one dimensional y vector, based on the size of training vector
car_y = np.ones(len(car_features))
non_car_y = np.zeros(len(non_car_features))
y = np.hstack((car_y, non_car_y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X_train)

# Apply the scaler to X
scaled__X_train = X_scaler.transform(X_train)

# Applying PCA
pca = PCA(n_components=1500)
pca.fit(scaled__X_train)

scaled__X_train = pca.transform(scaled__X_train)

scaled__X_test = X_scaler.transform(X_test)
scaled__X_test = pca.transform(scaled__X_test)

print("Feature vectors normalized")


print("Training feature vector dimension: ", scaled__X_train.shape)
print("Training y vector dimension: ",y_train.shape)
print("Test feature vector dimension: ", scaled__X_test.shape)
print("Test y vector dimension: ",y_test.shape)

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
pickle.dump(pca, open('pca.pkl', 'wb'))