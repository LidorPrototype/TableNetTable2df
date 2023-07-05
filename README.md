# TableNet using PyTorch

In this repo, you can have an implementation of the TableNet with Pytorch

## Goal

My goal is here to get a dataframe from an image, the image of a scanned document holding tabular data I will want to detect the image tables, crop the tables, and then extract the tabular data into a dataframe

## Data:

To populate the DummyDatabase folder you can refer to the following links:
 - [Marmot Datase](https://www.icst.pku.edu.cn/cpdp/docs/20190424190300041510.zip)
 - [Marmot Extended dataset](https://drive.google.com/drive/folders/1QZiv5RKe3xlOBdTzuTVuYRxixemVIODp)

## Model:

I will use a TableNet model with DenseNet121 as the main encoder.

I tried different encoders like VGG-19, ResNet, DenseNet121, efficientNet_B0, efficientNet and I got the best results with DenseNet121

> Note model itself is not uploaded because it's too big for GitHub uploads, you will be able to download the model in the future from this link. TK

## Model Predictions:

Predictions of the images in the folder DummyDatabase/test_images can be found in DummyDatabase/predictions

## Improvement idea:

The tables the model will detect and be any of the following:
1) Tables with full gridlines
2) Tables with only horizontal/vertical gridlines
3) Tables with only parts of horizontal/vertical gridlines
4) Tables without any gridlines drawn

So I had an idea which is, no matter what the table is of the above, remove all of the horizontal and vertical gridlines (if you find any), and then apply an OpenCV algorithm to detect the proper locations of all the gridlines and draw them artificially (The idea was implemented with help from StackOverflow).

You can find this idea implemented in the folder called GridlinesImprovement.

## Extract Tabular Data using `pytesseract`

Using the library `pytesseract` extract and process the tabular data and convert it into a dataframe.

_____________________________________________________________________________________________________________________________________

Author: Lidor ES
