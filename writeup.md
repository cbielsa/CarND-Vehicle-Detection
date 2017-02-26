##Writeup for the Vehicle Detection project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier,
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/exampleTrainingImages.jpg
[image2]: ./examples/carHogFeatures.jpg
[image3]: ./examples/noncarHogFeatures.jpg
[image4]: ./examples/carFeatureVector.jpg
[image5]: ./examples/noncarFeatureVector.jpg

[image6]: ./examples/sliding_windows.jpg
[image7]: ./examples/sliding_window.jpg
[image8]: ./examples/bboxes_and_heat.png
[image9]: ./examples/labels_map.png
[image10]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for image features extraction is in Section 1.1 of IPython notebook 'CarND-Vehicle-Detection.ipynb' (henceforth "the Notebook").

*Functions overview*
 * Function `readImage` takes an input an image file path and reads the image as RGB with pixel color channel intensities of type np.uint8 in the [0,255] range, regardless of whether the image is in jpg or png format.
 * Function `transform_colorSpace` transforms an RGB image to another color space.
 * Function `bin_spatial_features` transforms an image to a given pixel size and returns the intensities as a feature vector.
 * Function `color_hist_features` computes color histogram features for a given number of bins, in the input image color space.
 * Function `get_hog_features` computes a HOG features vector for given input parameterization, optionally returning as well a visualization. The low level computations are done by function `skimage.feature.hog`.
 * Function `extract_features_singleImage` is the highest-level feature-extraction function. It takes as input an RGB image and returns a feature vector including spatial, color histogram and HOG features. In particular, note that it is possible to generate HOG features for one, two or three color channels, depending on user input. The tuning selected for the processing pipeline is detailed in point 2 of this writeup.

*The dataset*

I collect `vehicle images from` two sources: the [GTI Vehicle database](https://www.gti.ssr.upm.es/data/Vehicle_database.html) and the [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/eval_tracking.php). Images from the former are extracted from short video clips, hence consecutive images are very similar to each other. Consecutive images from the latter, however, correspond to different vehicles or perspectives.

The dataset collection and training-test split is done in Section 3.1 of the Notebook.

I choose to split the dataset in 85% training and 15% test sets. For GTI images, I assign the first 85% files to the training set, and the remainder to the dataset. This way I avoid the problem of having very similar images in the training and test sets, which would have resulted on unreliably high test accuracy. For KITTI images, I randomize the split.

I collect `non-vehicle images` from GTI and the "Extras" directory provided by Udacity. In the "Extra" set, consecutive images are also similar to one another, hence I select the first 85% non-vehicle images of both GTI and "Extras" for the training set, and the remainder for the test set.

Next, I assign label `1` to `vehicle images` and `0` to `non-vehicle` images.

There is a total of 15,067 training and 2,693 test images, and both datasets are roughly balanced, with a similar number of images of either class. All images have 64x64 pixels, and I convert intensities to uint8 in range [0,255].

A random selection of images of either class are shown below:

![alt text][image1]

Since the vehicle detection pipeline uses a simple linear SVM as classifier (more on this later), the selection of a rich-enough feature space is fundamental for overall performance. This is in contrast to a deep CNN, whose initial layers can be trained to satisfactory perform feature extraction from raw RGB pixel intensities (as done in e.g. the [Behavioral Cloning project](https://github.com/cbielsa/CarND-BehavioralCloning-P3)).

I first explored color spaces in search for the space where images containing vehicles could be better differentiated. I also extracted HOG features with a variety of tunings (tunable parameters including number of orientation bins, pixels per cell, cells per block and color channels passed to the HOG feature extractor).

The figures below show color channels and HOG feature visualization for each channel for an example vehicle and non-vehicle image. This examples correspond to HSV color space, with HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]
![alt text][image3]

In addition to HOG features, color histograms can be used to provide color information, while spatial bin features provide both color and spatial context.

The plots below show the combined feature vectors for the same example vehicle and non-vehicle images displayed before, as returned by function `extract_features_singleImage`. Parameter selection is identical to the one used in the final processing pipeline. The order of the features is: first, spatial features; then, color histogram features; and, finally HOG features.

![alt text][image4]
![alt text][image5]

The y-axes are in logarithmic scale and, clearly, features have very different scales (spatial features are in [0,255] range, color histogram features can as large as order 1000, while HOG features are typically smaller than 1). Therefore, normalizatio will be performed at a later step.

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

