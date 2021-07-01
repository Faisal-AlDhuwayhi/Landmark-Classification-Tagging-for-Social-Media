# Landmark Classification & Tagging for Social Media 
In this project, you will apply the skills you have acquired in the Convolutional Neural Network (CNN) module to build a landmark classifier. You will take the first steps towards addressing this problem by building models to automatically predict the location of the image based on any landmarks depicted in the image. You will go through the machine learning design process end-to-end: performing data preprocessing, designing and training CNNs, comparing the accuracy of different CNNs, and using your own images to heuristically evaluate your best CNN.


## Project Steps
The high level steps of the project include:

1. *Create a CNN to Classify Landmarks (from Scratch)* - Here, you'll visualize the dataset, process it for training, and then build a convolutional neural network from scratch to classify the landmarks. You'll also describe some of your decisions around data processing and how you chose your network architecture.

2. *Create a CNN to Classify Landmarks (using Transfer Learning)* - Next, you'll investigate different pre-trained models and decide on one to use for this classification task. Along with training and testing this transfer-learned network, you'll explain how you arrived at the pre-trained network you chose.

3. *Write Your Landmark Prediction Algorithm* - Finally, you will use your best model to create a simple interface for others to be able to use your model to find the most likely landmarks depicted in an image. You'll also test out your model yourself and reflect on the strengths and weaknesses of your model.

The detailed project steps are included within the project notebook further above.


## Instructions
You need to install the following Python modules to complete this project:
- [cv2](https://docs.opencv.org/4.5.2/)
- [matplotlib](https://matplotlib.org/)
- [numpy](https://numpy.org/)
- [PIL](https://pillow.readthedocs.io/en/stable/)
- [torch](https://pytorch.org/)
- [torchvision](https://pytorch.org/vision/stable/index.html)

Note: If your code is taking too long to run, you will need to either reduce the complexity of your chosen CNN architecture or switch to running your code on a GPU. If you'd like to use a GPU, you can spin up an instance of your own.

## Dataset
The landmark images are a subset of the Google Landmarks Dataset v2. The dataset can be obtained using [this link](https://udacity-dlnfd.s3-us-west-1.amazonaws.com/datasets/landmark_images.zip).

You can find license information for the full dataset [on Kaggle](https://www.kaggle.com/google/google-landmarks-dataset).




