**Vehicle Detection**

The [Jupyter/IPython notebook](https://github.com/gardenermike/vehicle-tracking/blob/master/vehicle-detection.ipynb) in this repository uses an [SVM classifier from scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) to detect vehicles in images and video, building from examples in [Udacity's self-driving car nanodegree](https://www.udacity.com/drive).
I haven't used SVMs since I took Andrew Ng's course that eventually became the seed of Coursera. It certainly is easier now rather than writing it from scratch in Octave/MatLab.

This project represents more traditional machine learning: careful custom feature extraction and use of an older-style model rather than a deep convolutional classifier that would learn the features independently.

The detection procedure is, in summary:

* Extract [Histogram of Oriented Gradients (HOG)](http://www.learnopencv.com/histogram-of-oriented-gradients/) features as well as simplified spatial features and a color histogram from a set of car and not-car images
* Train a SVM classifier (with an [RBF kernel](https://en.wikipedia.org/wiki/Radial_basis_function_kernel)) on the extracted features
* Use a multi-scale sliding window to extract the same HOG/spatial/color features from relevant regions of an image to test with the trained classifier
* Threshold the matches found by the classifier using a heatmap to exclude outliers and false positives
* Draw bounding boxes around detected vehicles

I also combined my vehicle tracking with my [previous project](https://github.com/gardenermike/finding-lane-lines-reprise) that found lane boundaries in order to get a more complete tracking solution.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[car-and-hog]: ./examples/car-and-hog.jpg
[hog-sub]: ./examples/hog-sub.jpg
[boxed-1]: ./examples/boxed-1.png
[boxed-2]: ./examples/boxed-2.png
[car]: ./examples/53.png
[spatial]: ./examples/spatial_53.png
[histograms]: ./examples/histograms.png
[heatmap]: ./examples/heatmap.png
[video1]: ./project_video.mp4

## Feature Extraction

All of the feature extraction is performed by the `FeatureExtractor` class.

My training images came from a set of 8792 cars and 8968 non-car images, each of which were saved as 64x64 PNGs. The data is available from the [Udacity repository](https://github.com/udacity/CarND-Vehicle-Detection) that was the basis for this project.

Example "car" and "not car" images are below.

![Car and not-car][image1]

I began by trying out a variety of classifiers using a variety of colorspaces. I used a [LinearSVC model](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC), a [SVC model with an RBF kernel](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC), and a [Stochastic Gradient Descent SVM](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier) on the data. The RBF-kernel classifier consistently had greater than 99% accuracy on the test set, and was trainable on my machine, so I kept it for performance. The LinearSVC model would probably have been somewhat faster, at a cost in classification accuracy. The SGD model performed horribly for me: my accuracy consistently stayed at around 50% as I added epochs. I was hopeful that I could try it with data augmentation (which I experimented with), but I discovered quickly that an SVM does not have strong [translation invariance](https://en.wikipedia.org/wiki/Translational_symmetry) properties in the way a deep convolutional network does.

I also compared performance of various color spaces. The YCrCb space gave me the highest classification accuracy, around 99.5% on my test set. I suspect that the performance is related to the fact that the "Y" channel is basically grayscale, and so is insensitive to color differences betwen vehicles.


### Histogram of Oriented Gradients (HOG)

#### Extraction
Check out the `hog_features` method for details. Since most of the work is done by sklearn, there isn't a lot of code. I extract HOG features from the supplied list of channels (I used all three), flatten them, and concatenate them together.

The more interesting code is the sliding window implementation for the HOG features, detailed more below.

HOG features represent the direction the average gradient of a block is pointing in a cell. Intuitively, it takes the slope of the most dominant shape in each cell in a grid. An example image of a vehicle with HOG features is below.

![Car with HOG features][car-and-hog]

#### HOG Parameter choice

I experimented with HOG orientation count, pixels per cell, and cells per block, and the 8/8/2 values defaulted in the extractor for those values were a good balance between size and performance.

### Spatial features

"Spatial features" is a fancy way to say "low-resolution image." I downscale the 64x64 image to 16x16, leaving only the general suggestion of the shape. This blurring generalizes to "carness". I then flatten the resulting image, resulting in conceptually something more like a silhouette of a mountain range than a 2d grid of pixels.

An example 64x64 image and downsize 16x16 image are below. The 16x16 image preserves the general shape of the vehicle.

![Example car][car]
![Downsized car][spatial]

### Histogram of colors

I also extract a histogram of the colors in the image, with 32 bins. The idea is that cars will have a color distribution dissimilar from the background. An example color distribution for a vehicle in RGB is below.

![Color histograms][histograms]

### Combined features

I concatenate the HOG, spatial, and histogram features into one long vector to pass to the SVM.

### Training
In the third cell is my training code. I use my feature extractor to pull the features from all of the images, then use sklearn's [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) to normalize the full dataset. Normalization was important for a couple of reasons. First, it balances out any scale differences between the HOG, spatial, and color histogram features, and second, it ensures a consistent scale between examples. I use the same trained scaler later to normalize new features from the live data.
Also notable is that I split out 20% of the data as a test set to validate my classifier.
As mentioned above, I tried a variety of classifiers from sklearn and a variety of color spaces and other parameters to the feature extraction. I changed the defaults in my extractor to match the best values, and then used the defaults.
I found that the SVM classifier uses a lot of memory, so once I had the basics working I set it up on an EC2 spot instance to save time. The training went about 3x faster on a c4.2xlarge instance than on my old 2010 MacBook Pro.


### Sliding Window Search

The `find_cars` method started from Udacity sample code, but was customized to use the feature extractor class I built. It samples from a subsection of the image (from vertical pixel counts 400 to 656 of 720) and splits that section into an 8x8 pixel grid. Since HOG features are calculated in a grid, calculating them over the entire region and then subsampling is far more efficient than recalculating them for each subsection. The image is also scaled down for (much) faster performance. Switching from 2/3 scale to 1/2 scale cut the processing time in half, but lost too much information, so I stayed at 2/3 scale.

An overview image of the approach is below (credit goes to Udacity for this one):

![HOG subsampling][hog-sub]

Each sub-image sampled is scaled to 64x64 pixels to be tested as a car or not-car. If the image matches, its top-left and bottom-right corners are saved as a bounding box and returned from the function.


#### Results

Several example images with bounding boxes drawn by my classifier are in the notebook. A couple are included below. Note the overlapping boxes where multiple subregions were detected as a car.

![Detected vehicles 1][boxed-1]
![Detected vehicles 2][boxed-2]

---

### Video Implementation

My implementation on video added a couple of features, implemented in the `MovieProcessor` class.

#### Heatmap
In order to make sense of overlapping bounding boxes, I use a heatmap of the overlapping areas, find contiguous areas with the [label feature from scipy](https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.ndimage.measurements.label.html), and draw bounding boxes around the resulting areas.

![Example heatmap and bounding boxes][heatmap]

To leave out outlying data, I discard any data in the heatmap that does not have more than one overlapping box. Importantly, I keep the heatmap between frames, to allow a strengthening signal for a persistent object. To allow for change over time, however, I needed the second feature:

#### Decay
I decay the heatmap by 20% between each frame of video. I played around with the decay parameter a bit, and found that keeping 80% of the signal kept my boxes persistent without causing a "dragging" effect.


### Results

The resulting video is [in the repository](https://github.com/gardenermike/vehicle-tracking/blob/master/project_output.mp4) and also on [YouTube](https://youtu.be/8UL3n6VFNsc).

I also combined the vehicle tracking with the lane finding from a previous project to build a [combined video](https://github.com/gardenermike/vehicle-tracking/blob/master/project_video_output.mp4). ([YouTube link](https://youtu.be/qDGdRYXy1c0)) The combined video has a very interesting characteristic. As part of the lane finding pipeline, I corrected for camera distortion in the images. As mentioned before, the SVM classifier is not at all good with translation invariance, so correcting for camera distortion allowed for better vehicle detection. The difference is striking for the oncoming traffic. Numerous oncoming vehicles are tracked in the undistorted frames that were entirely missed without image correction.

---

### Discussion

When I first started tinkering with machine learning years ago, SVM seemed like a central technique, and what is now "deep learning" was not really on the radar. Things have changed! Now, with a variety of competing deep learning frameworks and advances in convolutional networks, SVMs seem kind of quaint and fragile. However, requiring only five minutes of training or less on a CPU for >99% accuracy is a big deal. Resource constraints still exist! Most computers are now mobile devices, without an onboard GPU and with limited battery life. An effective model that requires minimal resources has a lot of value.

The thing I discovered most in this project was the fragility of handcrafted feature extraction. This fragility was illustrated best to me by the differences in performance between the camera-calibrated video and the raw video. To my naked eye, the camera distortion is invisible without careful study. To the SVM, however, the difference is dramatic. Tiny rotations and distortions were enough to completely lose the oncoming traffic in the raw video. If I relied more on color histogram features than the hog features, I might do better, but I would expect numerous false positives from car-colored objects like billboards or even garbage.

I have a newfound respect for the power of deep convolutional networks, but also an awareness of their price in training time and dedicated GPU hardware.
