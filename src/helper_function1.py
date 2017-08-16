import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from skimage.feature import hog

from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import binary_dilation, grey_dilation
from moviepy.editor import VideoFileClip


def get_features():
    features = {}
    features['color_space'] = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    features['orient'] = 9  # HOG orientations
    features['pix_per_cell'] = 8 # HOG pixels per cell
    features['cell_per_block'] = 2 # HOG cells per block
    features['hog_channel'] = 0 # Can be 0, 1, 2, or "ALL"
    features['spatial_size'] = (16, 16) # Spatial binning dimensions
    features['hist_bins'] = 16    # Number of histogram bins
    features['spatial_feat'] = True # Spatial features on or off
    features['hist_feat'] = False # Histogram features on or off
    features['hog_feat'] = True # HOG features on or off
    features['y_start_stop'] = [450, 700] # Min and max in y to search in slide_window()
    return features


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(img_dirs):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    #for file in imgs:
    for img_dir in img_dirs:
        for img_file in os.listdir(img_dir):
            if '.png' not in img_file:
                continue
            # Read in each one by one
            image = mpimg.imread(img_dir + '/' + img_file)
            # apply color conversion if other than 'RGB'
            file_features = extract_features_for_one_image(image)

            features.append(file_features)

            flipped_image = np.fliplr(image)

            # apply color conversion if other than 'RGB'
            file_features = extract_features_for_one_image(flipped_image)

            features.append(file_features)

    # Return list of feature vectors
    return features


def extract_features_for_one_image(image):

    ## Set Parameter

    #color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    color_space = 'YCrCb'
    orient = 9  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
    #spatial_size = (16, 16) # Spatial binning dimensions
    spatial_size = (32, 32) # Spatial binning dimensions
    #hist_bins = 16    # Number of histogram bins
    hist_bins = 32
    spatial_feat = False # Spatial features on or off
    hist_feat = False # Histogram features on or off
    hog_feat = True # HOG features on or off


    file_features = []
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(image)      

    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        file_features.append(spatial_features)
    if hist_feat == True:
        # Apply color_hist()
        hist_features = color_hist(feature_image, nbins=hist_bins)
        file_features.append(hist_features)
    if hog_feat == True:
    # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        file_features.append(hog_features)

    return np.concatenate(file_features)
    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def extract_image_window(image, window):
    return image[window[0][1]:window[1][1], window[0][0]:window[1][0]]


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap < threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def read_pickled_models():
    clf = pickle.load(open('trained_classifier.pkl', 'rb'))
    X_scaler = pickle.load(open('x_scaler.pkl', 'rb'))
    pca = pickle.load(open('pca.pkl', 'rb'))
    return (clf, X_scaler, pca)

def dilate(img, times=3):
    for i in range(times):
        img = grey_dilation(img, size=(5,5))
    return img

def pipeline(img, clf, X_scaler, pca):
    y_start_stop = [350, 600] # Min and max in y to search in slide_window()

    #xy_windows = [[64,64], [128,128]]
    #xy_windows = [[128,128]]
    xy_windows = [[32,32], [64,64], [128,128], [256, 256]]

    xy_overlap = [0.8,0.8]
    windows = []
    for xy_window in xy_windows:
        tmp_window = slide_window(img, y_start_stop=y_start_stop, xy_window=xy_window, xy_overlap=xy_overlap)
        print("Temp window size ", len(tmp_window))
        windows = windows + tmp_window 

    selected_windows = []


    for window in windows:
        image_window = extract_image_window(img, window)
        # Resize it to 64x64
        image_window = cv2.resize(image_window, (64,64))
        image_features = extract_features_for_one_image(image_window)
        #print(image_features.shape)
        image_features = image_features.reshape(1, -1)
        scaled_features = X_scaler.transform(image_features)
        scaled_features = pca.transform(scaled_features)
        prediction = clf.predict(scaled_features)
        if prediction == 1:
            selected_windows.append(window)


    img_copy = img

    heat = np.zeros_like(img_copy[:,:,0]).astype(np.float)

    heat = add_heat(heat, selected_windows)
        
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 2)
    heat = dilate(heat, 3)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    structure = np.ones(9).reshape((3,3))   
    labels = label(heatmap, structure)
    draw_img = draw_labeled_bboxes(img_copy, labels)

    return draw_img

# Function to generate final video
def generate_video(clf, X_scaler, pca):
    input_video = '../test_video.mp4' 
    clip1 = VideoFileClip(input_video)
    output_file = '../output_project_video.mp4'
    output_clip = clip1.fl_image(lambda img: pipeline(img, clf, X_scaler, pca))
    output_clip.write_videofile(output_file, audio=False)