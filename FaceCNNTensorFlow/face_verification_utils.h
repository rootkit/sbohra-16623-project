// Copyright 2016 Sudev Bohra All rights reserved.
//
// Created by Sudev Bohra on 12/10/16.
//
// Not for commercial use.

#ifndef face_verification_utils_hpp
#define face_verification_utils_hpp

#include <stdio.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#import <opencv2/opencv.hpp>
#include "math.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/memmapped_file_system.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#endif /* face_verification_utils_hpp */

cv::Mat rotate(cv::Mat &src, double angle);
cv::Point2d transform(int x, int y, double ang, int org_rows, int org_cols, int new_rows, int new_cols);
double cosineSimilarity(float* a, float* b);
tensorflow::Tensor convertMatToTensor(cv::Mat &img);
