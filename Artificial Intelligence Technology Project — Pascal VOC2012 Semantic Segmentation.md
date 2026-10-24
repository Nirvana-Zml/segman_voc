# Artificial Intelligence Technology Project — Pascal VOC2012 Semantic Segmentation

# 1\. Project Introduction

In this project, you will make semantic segmentation predictions for images\. In order to do that, you need to build a semantic segmentation model, train on the Pascal VOC2012 dataset, and inference on the given test set\. In this process, you need to learn to load data, build models, train and validate models, build loss functions, etc\.

You will also need to write a report on what you learned during the project\. Submit your works as required before the deadline\. Any plagiarism or forgery will be considered as a "fail" for the project\. We have established a complete inspection process, please do not touch the red line\.

# 2\. Datasets

## 2\.1 Dataset 1 — Pascal VOC2012 \(Train / Validation\)

Download link: [http://host\.robots\.ox\.ac\.uk/pascal/VOC/voc2012/index\.html\#devkit](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit)

The development kit consists of the training/validation data, MATLAB code for reading the annotation data, support files, and example implementations for each competition\.

**Available downloads:**

- Download the training/validation data \(2GB tar file\)

- Download the development kit code and documentation \(500KB tar file\)

- Download the PDF documentation \(500KB PDF\)

- Browse the HTML documentation

- View the guidelines used for annotating the database \(VOC2011\)

- View the action guidelines used for annotating the action task images

**Segmentation set consists:**

- train set: 1465 images

- validation set: 1450 images \(optional\)

**Directory structure:** Annotations / ImageSets / JPEGImages / SegmentationClass

## 2\.2 Dataset 2 — Test Set \(Prediction\)

Dataset 2 is utilized as the test set to evaluate the model\.

- test set: 400 images \(extra set, not belonging to Pascal; only images, no label\)

# 3\. Tasks

1. Load segmentation data from the dataset, including images, labels, etc\.

2. Construct the semantic segmentation model\. Ensure that the input and output sizes of adjacent layers are compatible\.

3. Choose the appropriate loss function and build the backpropagation\.

4. Train and validate \(optional\) the model with Pascal VOC2012 segmentation train set and validate set \(optional\)\.

5. Save the weights of the model\.

6. Perform inference on the test set, and save the outputs which are single channel gray scale masks\.

7. The entire training process needs to be visualized \(use TensorBoard\)\.

8. Write the report\.

# 4\. Submission

## 4\.1 Project Report

e\.g\. `report_studentname_studentId.docx`, including:

1. The data processing and loading methods, the principle of model design, and the method of adjusting the parameters\.

2. Figure of network model structure\.

3. Figures for the accuracy curves and loss curves\.

## 4\.2 Source Code

e\.g\. `code_studentname_studentid`

## 4\.3 Test Set Predictions

e\.g\. `predict_studentname_studentid.zip`

## 4\.4 Packaging

Zip these three files into a folder named `studentname_studentid`, and submit the zipped folder to **2940986065@qq\.com** by **November 15th**\. Late submission leads to "fail" for the project\.

# 5\. Notes

- Final score is based on **mIoU performance** of the model on the test dataset and **quality of the experimental report**\.

- No plagiarism\. It is very easy to check whether the predictions are plagiarized\.

- Designing a new network model and using validation set to validate the model after each training epoch are encouraged\.
