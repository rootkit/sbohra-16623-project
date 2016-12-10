// Copyright 2016 Sudev Bohra All rights reserved.
//
// Created by Sudev Bohra on 12/10/16.
//
// Not for commercial use.

#import <AVFoundation/AVFoundation.h>
#import <UIKit/UIKit.h>

#include <memory>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/memmapped_file_system.h"

@interface FaceCNNViewController
    : UIViewController<UIGestureRecognizerDelegate,
                       AVCaptureVideoDataOutputSampleBufferDelegate> {
  IBOutlet UIView *previewView;
  IBOutlet UISegmentedControl *camerasControl;
  AVCaptureVideoPreviewLayer *previewLayer;
  AVCaptureVideoDataOutput *videoDataOutput;
  dispatch_queue_t videoDataOutputQueue;
  AVCaptureStillImageOutput *stillImageOutput;
  UIView *flashView;
  BOOL isUsingFrontFacingCamera;
  NSMutableArray *labelLayers;
  AVCaptureSession *session;
  std::unique_ptr<tensorflow::Session> tf_session;
  std::unique_ptr<tensorflow::MemmappedEnv> tf_memmapped_env;
  std::vector<std::string> labels;
}
@property(retain, nonatomic) CATextLayer *predictionTextLayer;
@property (nonatomic, strong) CIDetector *faceDetector;
- (IBAction)takePicture:(id)sender;
- (IBAction)switchCameras:(id)sender;

@end
