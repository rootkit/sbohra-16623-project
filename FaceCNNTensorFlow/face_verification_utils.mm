// Copyright 2016 Sudev Bohra All rights reserved.
//
// Created by Sudev Bohra on 12/10/16.
//
// Not for commercial use.

#include "face_verification_utils.h"


cv::Mat rotate(cv::Mat &src, double angle) {
    cv::Point2f center(src.cols/2.0, src.rows/2.0);
    
    cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
    // determine bounding rectangle
    cv::Rect bbox = cv::RotatedRect(center,src.size(), angle).boundingRect();
    // adjust transformation matrix
    rot.at<double>(0,2) += bbox.width/2.0 - center.x;
    rot.at<double>(1,2) += bbox.height/2.0 - center.y;
    
    cv::Mat dst;
    cv::warpAffine(src, dst, rot, bbox.size());
    return dst;
}

cv::Point2d transform(int x, int y, double ang, int org_rows, int org_cols, int new_rows, int new_cols) {
    double x0 = x - org_cols/2.0;
    double y0 = y - org_rows/2.0;
    double xx = x0*cos(ang) - y0*sin(ang) + new_cols/2.0;
    double yy = x0*cos(ang) + y0*sin(ang) + new_rows/2.0;
    cv::Point2d result(round(xx),round(yy));
    return result;
}

double cosineSimilarity(float* a, float* b) {
    double n = 0;
    double a2 = 0;
    double b2 = 0;
    for (int i = 0; i < 256; i++) {
        
        n = n + a[i]*b[i];
        a2 = a2 + a[i]*a[i];
        b2 = b2 + b[i]*b[i];
    }
    double d = sqrt(a2)*sqrt(b2);
    return n/d;
}

tensorflow::Tensor convertMatToTensor(cv::Mat &img) {
    int height = 128;
    int width = 128;
    int depth = 1;
    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT,
                                    tensorflow::TensorShape({1, height, width, depth}));
    
    auto input_tensor_mapped = input_tensor.tensor<float, 4>();
    
    const float* source_data = (float*)img.data;
    
    for (int y = 0; y < height; ++y) {
        const float* source_row = source_data + (y * width * depth);
        for (int x = 0; x < width; ++x) {
            const float* source_pixel = source_row + (x * depth);
            for (int c = 0; c < depth; ++c) {
                const float* source_value = source_pixel + c;
                input_tensor_mapped(0, y, x, c) = *source_value;
            }
        }
    }
    
    return input_tensor;
    
    
}
