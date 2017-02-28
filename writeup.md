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
[image2]: ./output_images/carHogFeatures.jpg
[image3]: ./output_images/noncarHogFeatures.jpg
[image4]: ./output_images/carFeatureVector.jpg
[image5]: ./output_images/noncarFeatureVector.jpg
[image6]: ./output_images/search_windows.jpg
[image7]: ./output_images/on_windows.jpg
[image8]: ./output_images/original_image.jpg
[image9]: ./output_images/test_images_processed
[image10]: ./output_images/pipeline_windows.png
[image11]: ./output_images/pipeline_heatmap.png
[image12]: ./output_images/pipeline_labels.png
[image13]: ./output_images/pipeline_final.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for image features extraction is in Section 1.1 of IPython notebook `CarND-Vehicle-Detection.ipynb` (henceforth `the Notebook`).

*Functions overview*
 * Function `readImage` takes an input an image file path and reads the image as RGB with pixel color channel intensities of type np.uint8 in the [0,255] range, regardless of whether the image is in jpg or png format.
 * Function `transform_colorSpace` transforms an RGB image to another color space.
 * Function `bin_spatial_features` transforms an image to a given pixel size and returns the intensities as a feature vector.
 * Function `color_hist_features` computes color histogram features for a given number of bins, in the input image color space.
 * Function `get_hog_features` computes a HOG features vector for given input parameterization, optionally returning as well a visualization. The low level computations are done by function `skimage.feature.hog`.
 * Function `extract_features_singleImage` is the highest-level feature-extraction function. It takes as input an RGB image and returns a feature vector including spatial, color histogram and HOG features. In particular, note that it is possible to generate HOG features for one, two or three color channels, depending on user input. The tuning selected for the processing pipeline is detailed in point 2 of this writeup.

*The dataset*

I collect `vehicle images from` two sources: the [GTI Vehicle database](https://www.gti.ssr.upm.es/data/Vehicle_database.html) and the [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/eval_tracking.php). Images from the former are extracted from short video clips, hence consecutive images are very similar to each other. Consecutive images from the latter, however, correspond to different vehicles or perspectives.

The dataset collection and training-test split is done in `Section 3.1` of the Notebook.

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

The y-axes are in logarithmic scale and, clearly, features have very different scales (spatial features are in `[0,255]` range, color histogram features can as large as order `1000`, while HOG features are typically smaller than `1`). Therefore, normalization will be performed at a later step.

####2. Explain how you settled on your final choice of HOG parameters.

`Section 4.2` in the Notebook contains the final choice of feature parameters, which is a trade-off between performance (as measured by classification accuracy in the test dataset) and processing time. In the final processing pipeline:
  * Images are converted to `HSV` color space.
  * HOG features are generated for channels S and V, but not for channel H.
  * For HOG feature extractions, `9 orientation bins`, `8 pixels per cell` and `2 cells per block` are used.
  * For spatial binning features, images in HSV are resized to `24x24 pixels`.
  * For histogram color features, `32 bins` are used.
  
This results in a feature vector size of `5352`, significantly smaller than the original image size of `64*64*3 = 12,288`.

For the exploration of color spaces, I used `Section 4.1` in the Notebook with a variety of car and non-car images. The `HSV` color space seems to be the one where car and non-car HOG features are better differentiated, particulary in the Saturation and Value channels.

To inform my choice of parameters for the HOG, spatial bin and color histogram feature selection, I used in the first place the experience gained from the exercises and experimentation in the Udacity nanodegree. For HOG parametrization, I also used HOG visualizations in `Section 4.1` of the Notebook.

The ultimate objective of the project is to implement a pipeline capable of reliably identifying vehicles in a video stream and, to a lesser extent, to do so in contained computational time. Therefore, I used classification accuracy of the trained SVM in the test set as an initial proxy for pipeline performance.

The selected parametrization achieved ~99% test accuracy with a moderate feature vector size. No other color space or parametrization resulted in better test accuracy with a similar number of parameters.

And although based only on test accuracy and performance of the feature selection/classifier in the test images, one by one, I initially decided not to use spatial bin features --thereby further reducing the number of features--, I later decided to introduce spatial features to improve the performance of the *dynamic* pipeline.

As it turns out, although the linear SVM classifier trained without spatial features returns fewer false positives, it also identifies fewer of the windows actually containing vehicles, and so in the dynamic context of the final pipeline where hotmaps are constructed taking into account detections in past recent frames, more false positives with more real detections (with spatial features) proved preferrable to fewer false positives with fewer real detections (without spatial features).

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

  * In `Section 4.2` of the Notebook, feature extraction is performed for the training and test datasets.
  * In `Section 4.3.`, normalization parameters are calculated based exclusively on the training dataset. That same normalization is then applied to both the training and the test datasets. Note that it is important to calculate the normalization parameters without the test set to avoid data leakage and ensure that test accuracy provides a performance metric realistic when the classifier is applied to real-world data. For the normalization, I use ´sklearn.preprocessing.StandardScaler()´.
  * Next, in `Section 4.4` I randomly shuffle the normalized training dataset. The same shuffling is indeed applied to features vectors and labels.
  * Finally, in `Section 4.5` I construct and train a SVM classifier with a linear kernel on the shuffled normalized features of the training set. I use `sklearn.svm.linearSVC` with `C=1e-4` and `dual=False`. That choice of the misclassification penalty `C` resulted in a maximum test accuracy of 98.8%, by reducing overfitting relative to the default value `C=1`. On the other hand, the [scikit](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) documentation recommends to solve the primal optimization problem when n_samples>n_features, as it is our case (`n_training_samples = 15,067`, `n_features = 5,352`).

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

`Section 2.2` of the Notebook implements all functions related to sliding window search.

*Functions overview*
  * Function `draw_boxes` draws the input boxes on the input image.
  * Function `slide_window` generates a list of search windows of a single given size within a given image region and with given overlap.
  * Function `slide_windows_multiScale` generates a list of windows at different scales by calling `slide_window` for different window sizes and overlap values. This is the function called at each processing cycle to generate search windows.
  * Function `search_windows` takes an input image (the video frame) and a list of search windows, and returns a list with the windows likely to contain a vehicle. To do this, the function loops over the input search windows, and for each of them: resizes the part of the image contained in the window to 64x64 pixels, extracts features, normalizes them, performs inference with the linear SVM classifier and, if the window is labeled as containing a vehicle, adds that windows the the output list. Indeed, the feature extraction and normalization methods are identical to those used to train the classifier.
  * Function `createHeatmap` takes as input a list of windows thought to contain vehicles (possibly collected over the last `iNumFramesTracked` cycles) and generates a detection heatmap with the same width and height as the video frame. The most likely a pixel is to belong to a vehicle, the higher its value in the heatmap.
  * Function `apply_threshold` applies an input threshold to the input heatmap, such that elements with a value smaller than the threshold are set to zero. It also clips values to the [0,255] range, so that the heatmap can be plotted with plt.imshow().
  * Function `compute_and_draw_labeled_bboxes` takes as input the original image, a thresholded detection heatmap, and size and height-to-width window validity thresholds, and returns the original images with detection boundary boxes drawn on it. To do this, it first isolates each separated region in the heatmap using function `scipy.ndimage.measurements.label.labels()`, it then defines a boundary box for each region based on min and max x- and y-values, and finally, it draws the boundary boxes with dimensions and height-to-width ratio within expected thresholds, to help filter out false positives.

The following figures show the original test image `test1.jpg`, followed by the same image with all search windows returned by `slide_windows_multiScale` drawn on it, with a different color for each set of window sizes:

![alt text][image8]
![alt text][image6]

As seen in the image above, for the processing of the project video, I implemented sliding windows of three sizes:
  * 128x128 pixels, with overlap 0.8.
  * 96x96 pixels, with overlap 0.9.
  * 64x64 pixels, with overlap 0.8.
  
This amounted to 1047 windows searched per image.
Originally, I also implemented larger windows on the bottom-most section of the image, but they proved to be of no use in the project video and so I decided to remove them to reduce processing time. For a general-purpose pipeline capable to detect vehicles in the near vicinities, however, I would have included windows of 192x192 or 256x256 pixels in the bottom-most section of the image.

As per the choice of overlaps, I simply increased them until the detection algorithm proved to work well enough in all the test images. The results on the test images are indeed encouraging, with e.g. 70 windows containing (parts of) a vehicle detected in `test1.jpg`, as shown below:

![alt text][image7]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_withVehicleDetection.mp4)


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

