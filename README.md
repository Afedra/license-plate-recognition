# LICENCE PLATE RECOGNITION SYSTEM #

[![Travis](https://travis-ci.org/kapscapital/LicensePlateRecognition.png)](https://travis-ci.org/kapscapital/LicensePlateRecognition)
[![circleci](https://circleci.com/gh/kapscapital/LicensePlateRecognition.png)](https://circleci.com/gh/kapscapital/LicensePlateRecognition)

* [Demo](http://alrp.kapscapital.com/) Try Demo

## **About**
A python program that uses concepts of image processing and OCR to identify the characters on a license plate. The OCR aspect was done with machine learning.

## **Functionality**
1. A Web GUI interface that makes image selection easier
2. Performs all the stages of Automatic License plate recognition (ALPR); plate localization, character segmentation and character recognition
3. Saves the license plate characters in the database
4. You can generate your model that will be used by the ALPR
5. You can compare the performance of supervised learning classifiers
6. You can use your own training data
7. Easy visualization for debugging purposes

## **Dependencies**
The program was written with python 2.7 and the following python packages are required
* [Numpy](http://docs.scipy.org/doc/numpy-1.10.0) Numpy is a python package that helps in handling n-dimensional arrays and matrices
* [Scipy](http://scipy.org) Scipy for scientific python
* [Scikit-image](http://scikit-image.org/) Scikit-image is a package for image processing
* [Scikit-learn](http://scikit-learn.org/) Scikit-learn is for all machine learning operations
* [Matplotlib](http://matplotlib.org) Matplotlib is a 2D plotting library for python

## **How to use**
1. Clone the repository or download the zip `git clone https://github.com/kapscapital/LicensePlateRecognition`
2. Change to the cloned directory (or extracted directory)
3. Create a virtual environment with virtualenv or virtualenvwrapper
4. Install all the necessary dependencies by using pip `pip install -r requirements.txt`
5. Start the program `./generate.sh`

## **Other Information**
- For windows users, you may need to install BLAS/LAPACK before you can install `scipy`
