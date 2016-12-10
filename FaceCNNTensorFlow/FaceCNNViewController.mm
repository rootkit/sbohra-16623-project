// Copyright 2016 Sudev Bohra All rights reserved.
//
// Created by Sudev Bohra on 12/10/16.
//
// Not for commercial use.

#import <AssertMacros.h>
#import <AssetsLibrary/AssetsLibrary.h>
#import <CoreImage/CoreImage.h>
#import <ImageIO/ImageIO.h>
#import "FaceCNNViewController.h"

#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#import <opencv2/opencv.hpp>


#include <sys/time.h>
#import <foundation/foundation.h>

#include "ios_image_load.h"
#include "tensorflow_utils.h"
#include "face_verification_utils.h"

#include <chrono>
#include <thread>


using Clock = std::chrono::steady_clock;
using std::chrono::time_point;
using std::chrono::duration_cast;
using std::chrono::milliseconds;


const int wanted_input_width = 128;
const int wanted_input_height = 128;
const int wanted_input_channels = 1;
const float input_mean = 0.0;
const float input_std = 255.0f;
const std::string input_layer_name = "Placeholder";
const std::string output_layer_name = "mfm_fc1";
static NSString* model_file_name = @"frozen_optimized_graph";
static NSString* model_file_type = @"pb";

const bool model_uses_memory_mapping = false;



static const NSString *AVCaptureStillImageIsCapturingStillImageContext =
@"AVCaptureStillImageIsCapturingStillImageContext";

@interface FaceCNNViewController (InternalMethods)
- (void)setupAVCapture;
- (void)teardownAVCapture;

@end



@implementation FaceCNNViewController{
    dlib::shape_predictor sp;
    NSArray<AVMetadataObject*>* currentRectangles;
    float* result;
    float* lockFace;
    cv::Mat storeImg;
    tensorflow::Tensor tensor;
    IBOutlet UILabel *A;
    int countFail;
    int countSuccess;
}


@synthesize faceDetector = _faceDetector;


time_point<Clock> lastRunCNN = Clock::now();

using tensorflow::uint8;


- (void) clearCounts {
    countFail = 0;
    countSuccess = 0;
}

bool dlib_on = false;

- (float*) process: (int) ind {
    NSString* image_path;
    
    if (ind < 100) {
        if (ind < 10) {
            
            image_path = FilePathForResourceName([NSString stringWithFormat:@"Ana_Guevara/Ana_Guevara_000%d",ind], @"jpg");
        } else {
            image_path = FilePathForResourceName([NSString stringWithFormat:@"Ana_Guevara/Ana_Guevara_00%d",ind], @"jpg");
            
        }
    } else {
        ind = ind - 100;
        if (ind < 10) {
            
            image_path = FilePathForResourceName([NSString stringWithFormat:@"Hillary_Clinton/Hillary_Clinton_000%d",ind], @"jpg");
        } else {
            image_path = FilePathForResourceName([NSString stringWithFormat:@"Hillary_Clinton/Hillary_Clinton_00%d",ind], @"jpg");
            
        }
    }
    
    
    
    
    
    LOG(INFO) << image_path;
    int image_width;
    int image_height;
    int image_channels;
    std::vector<tensorflow::uint8> image_data = LoadImageFromFile([image_path UTF8String], &image_width, &image_height, &image_channels);
    const int wanted_width = 128;
    const int wanted_height = 128;
    const int wanted_channels = 1;
    const float input_mean = 0;//117.0f;
    const float input_std = 255.0f;
    assert(image_channels >= wanted_channels);
    tensorflow::Tensor image_tensor(
                                    tensorflow::DT_FLOAT,
                                    tensorflow::TensorShape({
        1, wanted_height, wanted_width, wanted_channels}));
    auto image_tensor_mapped = image_tensor.tensor<float, 4>();
    tensorflow::uint8* in = image_data.data();
    tensorflow::uint8* in_end = (in + (image_height * image_width * image_channels));
    float* out = image_tensor_mapped.data();
    for (int y = 0; y < wanted_height; ++y) {
        const int in_y = (y * image_height) / wanted_height;
        tensorflow::uint8* in_row = in + (in_y * image_width * image_channels);
        float* out_row = out + (y * wanted_width * wanted_channels);
        for (int x = 0; x < wanted_width; ++x) {
            const int in_x = (x * image_width) / wanted_width;
            tensorflow::uint8* in_pixel = in_row + (in_x * image_channels);
            Float32* out_pixel = out_row + (x * wanted_channels);
            for (int c = 0; c < wanted_channels; ++c) {
                out_pixel[c] = (in_pixel[c] - input_mean) / input_std;
            }
        }
    }
    tensor = image_tensor;
    
    std::string input_layer = "Placeholder";
    std::string output_layer = "mfm_fc1";
    std::vector<tensorflow::Tensor> outputs;
    time_point<Clock> start = Clock::now();
    tensorflow::Status run_status = self->tf_session->Run({{input_layer, image_tensor}},
                                                          {output_layer}, {}, &outputs);
    
    time_point<Clock> end = Clock::now();
    milliseconds diff = duration_cast<milliseconds>(end - start);
    std::cout << "Timing:"<< diff.count() << "ms" << std::endl;
    
    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status;
        
        return NULL;
    }
    tensorflow::string status_string = run_status.ToString();
    
    
    tensorflow::Tensor* output = &outputs[0];
    
    auto f = output->flat<float>();

    float* new_arr = new float[256]();
    const int count = f.size();
    for (int i = 0; i < count; ++i) {
        const float value = f(i);
        new_arr[i] = value;
        LOG(INFO) << "Value " << value;
    }
    
    
    
    return new_arr;
}


- (void)setupAVCapture {
    
    NSError *error = nil;
    
    session = [AVCaptureSession new];
    if ([[UIDevice currentDevice] userInterfaceIdiom] ==
        UIUserInterfaceIdiomPhone)
        [session setSessionPreset:AVCaptureSessionPreset640x480];
    else
        [session setSessionPreset:AVCaptureSessionPresetPhoto];
    
    AVCaptureDevice *device =
    [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    AVCaptureDeviceInput *deviceInput =
    [AVCaptureDeviceInput deviceInputWithDevice:device error:&error];
    assert(error == nil);
    
    isUsingFrontFacingCamera = NO;
    if ([session canAddInput:deviceInput]) [session addInput:deviceInput];
    
    stillImageOutput = [AVCaptureStillImageOutput new];
    [stillImageOutput
     addObserver:self
     forKeyPath:@"capturingStillImage"
     options:NSKeyValueObservingOptionNew
     context:(void *)(AVCaptureStillImageIsCapturingStillImageContext)];
    if ([session canAddOutput:stillImageOutput])
        [session addOutput:stillImageOutput];
    
    videoDataOutput = [AVCaptureVideoDataOutput new];
    
    NSDictionary *rgbOutputSettings = [NSDictionary
                                       dictionaryWithObject:[NSNumber numberWithInt:kCMPixelFormat_32BGRA]
                                       forKey:(id)kCVPixelBufferPixelFormatTypeKey];
    [videoDataOutput setVideoSettings:rgbOutputSettings];
    [videoDataOutput setAlwaysDiscardsLateVideoFrames:YES];
    
    auto conn = [videoDataOutput connectionWithMediaType:AVMediaTypeVideo];
    [conn setEnabled:YES];
    conn.videoOrientation = AVCaptureVideoOrientationPortrait;
    
    videoDataOutputQueue =
    dispatch_queue_create("VideoDataOutputQueue", DISPATCH_QUEUE_SERIAL);
    [videoDataOutput setSampleBufferDelegate:self queue:videoDataOutputQueue];
    
    
    if ([session canAddOutput:videoDataOutput])
        [session addOutput:videoDataOutput];
    

    
    
    previewLayer = [[AVCaptureVideoPreviewLayer alloc] initWithSession:session];
    [previewLayer setBackgroundColor:[[UIColor yellowColor] CGColor]];
    [previewLayer setVideoGravity:AVLayerVideoGravityResizeAspect];
    CALayer *rootLayer = [previewView layer];
    [rootLayer setMasksToBounds:YES];
    [previewLayer setFrame:[rootLayer bounds]];
    [rootLayer addSublayer:previewLayer];
    [self switchCameras: nil];
    [session startRunning];
    
    [session release];
    if (error) {
        UIAlertView *alertView = [[UIAlertView alloc]
                                  initWithTitle:[NSString stringWithFormat:@"Failed with error %d",
                                                 (int)[error code]]
                                  message:[error localizedDescription]
                                  delegate:nil
                                  cancelButtonTitle:@"Dismiss"
                                  otherButtonTitles:nil];
        [alertView show];
        [alertView release];
        [self teardownAVCapture];
    }
}


- (void)teardownAVCapture {
    [videoDataOutput release];
    if (videoDataOutputQueue) dispatch_release(videoDataOutputQueue);
    [stillImageOutput removeObserver:self forKeyPath:@"isCapturingStillImage"];
    [stillImageOutput release];
    [previewLayer removeFromSuperlayer];
    [previewLayer release];
}

- (void)observeValueForKeyPath:(NSString *)keyPath
                      ofObject:(id)object
                        change:(NSDictionary *)change
                       context:(void *)context {
    if (context == AVCaptureStillImageIsCapturingStillImageContext) {
        BOOL isCapturingStillImage =
        [[change objectForKey:NSKeyValueChangeNewKey] boolValue];
        
        if (isCapturingStillImage) {
            // do flash bulb like animation
            flashView = [[UIView alloc] initWithFrame:[previewView frame]];
            [flashView setBackgroundColor:[UIColor whiteColor]];
            [flashView setAlpha:0.f];
            [[[self view] window] addSubview:flashView];
            
            [UIView animateWithDuration:.4f
                             animations:^{
                                 [flashView setAlpha:1.f];
                             }];
        } else {
            [UIView animateWithDuration:.4f
                             animations:^{
                                 [flashView setAlpha:0.f];
                             }
                             completion:^(BOOL finished) {
                                 [flashView removeFromSuperview];
                                 [flashView release];
                                 flashView = nil;
                             }];
        }
    }
}

- (AVCaptureVideoOrientation)avOrientationForDeviceOrientation:
(UIDeviceOrientation)deviceOrientation {
    AVCaptureVideoOrientation vresult =
    (AVCaptureVideoOrientation)(deviceOrientation);
    if (deviceOrientation == UIDeviceOrientationLandscapeLeft)
        vresult = AVCaptureVideoOrientationLandscapeRight;
    else if (deviceOrientation == UIDeviceOrientationLandscapeRight)
        vresult = AVCaptureVideoOrientationLandscapeLeft;
    return vresult;
}

- (IBAction)takePicture:(id)sender {
    if ([session isRunning]) {
        [session stopRunning];
        
        
        
        [sender setTitle:@"🔒" forState:UIControlStateNormal];
        
        flashView = [[UIView alloc] initWithFrame:[previewView frame]];
        [flashView setBackgroundColor:[UIColor whiteColor]];
        [flashView setAlpha:0.f];
        [[[self view] window] addSubview:flashView];
        
        [UIView animateWithDuration:.2f
                         animations:^{
                             [flashView setAlpha:1.f];
                         }
                         completion:^(BOOL finished) {
                             [UIView animateWithDuration:.2f
                                              animations:^{
                                                  [flashView setAlpha:0.f];
                                              }
                                              completion:^(BOOL finished) {
                                                  [flashView removeFromSuperview];
                                                  [flashView release];
                                                  flashView = nil;
                                              }];
                         }];
        
    } else {
        lockFace = result;
        LOG(INFO) << "Saved lockFace " << lockFace << std::endl;
        result = new float[256]();
        LOG(INFO) << "Saved lockFace " << lockFace << std::endl;
        [session startRunning];
        [sender setTitle:@"😄🔓" forState:UIControlStateNormal];
    }
}

+ (CGRect)videoPreviewBoxForGravity:(NSString *)gravity
                          frameSize:(CGSize)frameSize
                       apertureSize:(CGSize)apertureSize {
    CGFloat apertureRatio = apertureSize.height / apertureSize.width;
    CGFloat viewRatio = frameSize.width / frameSize.height;
    
    CGSize size = CGSizeZero;
    if ([gravity isEqualToString:AVLayerVideoGravityResizeAspectFill]) {
        if (viewRatio > apertureRatio) {
            size.width = frameSize.width;
            size.height =
            apertureSize.width * (frameSize.width / apertureSize.height);
        } else {
            size.width =
            apertureSize.height * (frameSize.height / apertureSize.width);
            size.height = frameSize.height;
        }
    } else if ([gravity isEqualToString:AVLayerVideoGravityResizeAspect]) {
        if (viewRatio > apertureRatio) {
            size.width =
            apertureSize.height * (frameSize.height / apertureSize.width);
            size.height = frameSize.height;
        } else {
            size.width = frameSize.width;
            size.height =
            apertureSize.width * (frameSize.width / apertureSize.height);
        }
    } else if ([gravity isEqualToString:AVLayerVideoGravityResize]) {
        size.width = frameSize.width;
        size.height = frameSize.height;
    }
    
    CGRect videoBox;
    videoBox.size = size;
    if (size.width < frameSize.width)
        videoBox.origin.x = (frameSize.width - size.width) / 2;
    else
        videoBox.origin.x = (size.width - frameSize.width) / 2;
    
    if (size.height < frameSize.height)
        videoBox.origin.y = (frameSize.height - size.height) / 2;
    else
        videoBox.origin.y = (size.height - frameSize.height) / 2;
    
    return videoBox;
}

- (NSNumber *) exifOrientation: (UIDeviceOrientation) orientation
{
    int exifOrientation;

    
    enum {
        PHOTOS_EXIF_0ROW_TOP_0COL_LEFT			= 1, //   1  =  0th row is at the top, and 0th column is on the left (THE DEFAULT).
        PHOTOS_EXIF_0ROW_TOP_0COL_RIGHT			= 2, //   2  =  0th row is at the top, and 0th column is on the right.
        PHOTOS_EXIF_0ROW_BOTTOM_0COL_RIGHT      = 3, //   3  =  0th row is at the bottom, and 0th column is on the right.
        PHOTOS_EXIF_0ROW_BOTTOM_0COL_LEFT       = 4, //   4  =  0th row is at the bottom, and 0th column is on the left.
        PHOTOS_EXIF_0ROW_LEFT_0COL_TOP          = 5, //   5  =  0th row is on the left, and 0th column is the top.
        PHOTOS_EXIF_0ROW_RIGHT_0COL_TOP         = 6, //   6  =  0th row is on the right, and 0th column is the top.
        PHOTOS_EXIF_0ROW_RIGHT_0COL_BOTTOM      = 7, //   7  =  0th row is on the right, and 0th column is the bottom.
        PHOTOS_EXIF_0ROW_LEFT_0COL_BOTTOM       = 8  //   8  =  0th row is on the left, and 0th column is the bottom.
    };
    
    switch (orientation) {
        case UIDeviceOrientationPortraitUpsideDown:  // Device oriented vertically, home button on the top
            exifOrientation = PHOTOS_EXIF_0ROW_LEFT_0COL_BOTTOM;
            break;
        case UIDeviceOrientationLandscapeLeft:       // Device oriented horizontally, home button on the right
            if (isUsingFrontFacingCamera)
                exifOrientation = PHOTOS_EXIF_0ROW_BOTTOM_0COL_RIGHT;
            else
                exifOrientation = PHOTOS_EXIF_0ROW_TOP_0COL_LEFT;
            break;
        case UIDeviceOrientationLandscapeRight:      // Device oriented horizontally, home button on the left
            if (isUsingFrontFacingCamera)
                exifOrientation = PHOTOS_EXIF_0ROW_TOP_0COL_LEFT;
            else
                exifOrientation = PHOTOS_EXIF_0ROW_BOTTOM_0COL_RIGHT;
            break;
        case UIDeviceOrientationPortrait:            // Device oriented vertically, home button on the bottom
        default:
            exifOrientation = PHOTOS_EXIF_0ROW_RIGHT_0COL_TOP;
            break;
    }
    return [NSNumber numberWithInt:exifOrientation];
}

- (void)captureOutput:(AVCaptureOutput *)captureOutput
didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
       fromConnection:(AVCaptureConnection *)connection {
   
    connection.videoOrientation = AVCaptureVideoOrientationPortrait;
    CVPixelBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    CVPixelBufferLockBaseAddress(imageBuffer, 0);

    CIImage *ciImage = [CIImage imageWithCVPixelBuffer:imageBuffer];

    int height = CVPixelBufferGetHeight(imageBuffer);
    
    
    CGAffineTransform transform = CGAffineTransformMakeScale(1, -1);
    transform = CGAffineTransformTranslate(transform,0, -height);
    
    UIDeviceOrientation curDeviceOrientation = [[UIDevice currentDevice] orientation];
    
    NSDictionary *imageOptions = nil;
    
    imageOptions = [NSDictionary dictionaryWithObject:[self exifOrientation:curDeviceOrientation]
                                               forKey:CIDetectorImageOrientation];
    
    NSArray<CIFeature*>*features = [self.faceDetector featuresInImage:ciImage];
    
    std::vector<dlib::rectangle> convertedRectangles;
    std::vector<std::vector<dlib::point>> facePoints;
    for ( CIFaceFeature *ff in features ) {
        // find the correct position for the square layer within the previewLayer
        // the feature box originates in the bottom left of the video frame.
        // (Bottom right if mirroring is turned on)
        const CGRect rect = CGRectApplyAffineTransform(ff.bounds, transform);
        long left = rect.origin.x;
        long top = rect.origin.y;
        long right = left + rect.size.width;
        long bottom = top + rect.size.height;
        
        
        dlib::rectangle dlibRect(left, top, right, bottom);
        
        
        if (ff.hasLeftEyePosition && ff.hasRightEyePosition && ff.hasMouthPosition) {
            convertedRectangles.push_back(dlibRect);
            auto lefteye = CGPointApplyAffineTransform(ff.leftEyePosition, transform);
            auto righteye = CGPointApplyAffineTransform(ff.rightEyePosition, transform);
            auto mouth = CGPointApplyAffineTransform(ff.mouthPosition, transform);
            dlib::point leye(lefteye.x, lefteye.y);
            dlib::point reye(righteye.x, righteye.y);
            dlib::point m(mouth.x, mouth.y);
            std::vector<dlib::point> facePoint;
            facePoint.push_back(leye);
            facePoint.push_back(reye);
            facePoint.push_back(m);
            facePoints.push_back(facePoint);
        }
        
    }
    
   
    
    dlib::array2d<dlib::bgr_pixel> img;
    
    
    CVPixelBufferLockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
    
    size_t width = CVPixelBufferGetWidth(imageBuffer);
    
    
    
    std::cout << "HEIGHT BRO " << height << " " << width << std::endl;
    
    //size_t height = CVPixelBufferGetHeight(imageBuffer);
    char *baseBuffer = (char *)CVPixelBufferGetBaseAddress(imageBuffer);
    
    // set_size expects rows, cols format
    img.set_size(height, width);
    
    // copy samplebuffer image data into dlib image format
    img.reset();
    long position = 0;
    while (img.move_next()) {
        dlib::bgr_pixel& pixel = img.element();
        
        // assuming bgra format here
        long bufferLocation = position * 4; //(row * width + column) * 4;
        char b = baseBuffer[bufferLocation];
        char g = baseBuffer[bufferLocation + 1];
        char r = baseBuffer[bufferLocation + 2];
        //        we do not need this
        //        char a = baseBuffer[bufferLocation + 3];
        
        dlib::bgr_pixel newpixel(b, g, r);
        pixel = newpixel;
        
        position++;
    }
    
    CVPixelBufferUnlockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
    cv::Mat cvImage = dlib::toMat(img);
    
    
    // for every detected face
    for (unsigned long j = 0; j < convertedRectangles.size(); ++j)
    {
        
        if (dlib_on) {
            dlib::rectangle oneFaceRect = convertedRectangles[j];
            
            // detect all landmarks
            dlib::full_object_detection shape = sp(img, oneFaceRect);
            
            
            
            long leye_x = 0;
            long leye_y = 0;
            
            for (unsigned long l = 37; l < 42; l++ ) {
                if (l == 39) {
                    continue;
                }
                dlib::point c = shape.part(l);
                leye_x += c.x();
                leye_y += c.y();
            }
            leye_x = leye_x/4.0;
            leye_y = leye_y/4.0;
            
            long reye_x = 0;
            long reye_y = 0;
            
            for (unsigned long m = 43; m < 48; m++ ) {
                if (m == 45) {
                    continue;
                }
                dlib::point c = shape.part(m);
                
                reye_x += c.x();
                reye_y += c.y();
            }
            reye_x = reye_x/4.0;
            reye_y = reye_y/4.0;
            
            dlib::point leye(leye_x, leye_y);
            dlib::point reye(reye_x, reye_y);
            dlib::point lmouth = shape.part(48);
            dlib::point rmouth = shape.part(54);
            dlib::point nose = shape.part(30);
            
            dlib::point arr[] = {leye, reye, nose, lmouth, rmouth};
            std::vector<dlib::point> f5pt (arr, arr + sizeof(arr) / sizeof(arr[0]) );
            
    
            align(cvImage, f5pt,128, 48, 40);
            
        } else {
            storeImg = align2(cvImage, facePoints[j],128, 48, 40);
        }
        
        int kernel_size = 3;
        int scale = 1;
        int delta = 0;
        cv::Mat dst;
        cv::Mat mean;
        cv::Mat sd;
        cv::Laplacian( storeImg, dst, CV_8U);
        cv::meanStdDev(dst, mean, sd);
        auto stddev = sd.at<double>(0,0);
        LOG(INFO) << "sd is " << sd.at<double>(0,0);
        //cv::Mat im_gray;
        if (stddev < 10.0) {
            LOG(INFO) << "sd too low " << stddev << std::endl;
            return;
        } else if (storeImg.rows != 128 || storeImg.cols != 128 ) {
            LOG(INFO) << "crop too small" << storeImg.rows << " " << storeImg.cols << std::endl;
            return;
        }
        
        [self runCNNOnFrame:storeImg];
        //        long l = oneFaceRect.width();
        //        cv::Size size(l,l);//the dst image size,e.g.100x100
        //        cv::Mat jImage;//dst image
        //        cv::resize(self.joint,jImage,size);//resize image
        //        dlib::array2d<dlib::rgb_pixel> imgrgb;
        //        cv::cvtColor(jImage, jImage, CV_RGB2BGRA);
        //        if (x+jImage.cols < cvImage.cols && y+jImage.rows < cvImage.rows) {
        ////            cv::cvtColor(jImage, jImage, CV_RGB2BGR);
        ////            jImage.copyTo(cvImage(cv::Rect((int)x,(int)y,jImage.cols, jImage.rows)));
        //            cv::Point2i p = cv::Point2i(x,y-100);
        //            overlayImage(cvImage, jImage,
        //                         cvImage, p);
        //        }
        
        //cv::Mat output;
        //cv::Point2i nose = cv::Point2i(x-l/2,y-l);
        //cv::Point2i p = cv::Point2i(reye_x,reye_y);
        //        overlayImage(cvImage, jImage,
        //                     cvImage, p);
        //overlayImage(cvImage, aligned, cvImage, p);
        //
        //
        // dlib::cv_image<dlib::bgr_pixel> cvimg =dlib::cv_image<dlib::bgr_pixel>(cvImage);
        // assign_image(img, cvimg);
        
        
        
        
        
        //        if (self.tf_session != nullptr) {
        //            std::string input_layer = "Placeholder";
        //            std::string output_layer = "mfm_fc1";
        //            std::vector<tensorflow::Tensor> outputs;
        //            time_point<Clock> start = Clock::now();
        //            tensorflow::Status run_status = self.tf_session->Run({{input_layer, input_tensor}},
        //                                                                 {output_layer}, {}, &outputs);
        //
        //            time_point<Clock> end = Clock::now();
        //            milliseconds diff = duration_cast<milliseconds>(end - start);
        //            std::cout << "Timing:"<< diff.count() << "ms" << std::endl;
        //
        //            if (!run_status.ok()) {
        //                LOG(ERROR) << "Running model failed: " << run_status;
        //                tensorflow::LogAllRegisteredKernels();
        //
        //            }
        //            tensorflow::string status_string = run_status.ToString();
        //
        //            tensorflow::Tensor* output = &outputs[0];
        //        }
        //
        
        
        
        
        
        //dlib::point z(x,y);
        //draw_solid_circle(img, z, 10, dlib::rgb_pixel(0, 255, 255));
        
        //self.center = z;
        
        //std::cout << x << y;
        
        
        // lets put everything back where it belongs
        //    CVPixelBufferLockBaseAddress(imageBuffer, 0);
        //
        //    // copy dlib image data back into samplebuffer
        //    img.reset();
        //    position = 0;
        //    while (img.move_next()) {
        //        dlib::bgr_pixel& pixel = img.element();
        //
        //        // assuming bgra format here
        //        long bufferLocation = position * 4; //(row * width + column) * 4;
        //        baseBuffer[bufferLocation] = pixel.blue;
        //        baseBuffer[bufferLocation + 1] = pixel.green;
        //        baseBuffer[bufferLocation + 2] = pixel.red;
        //        //        we do not need this
        //        //        char a = baseBuffer[bufferLocation + 3];
        //
        //        position++;
        //    }
        //    CVPixelBufferUnlockBaseAddress(imageBuffer, 0);
        // unlock buffer again until we need it again
        
        //CVPixelBufferUnlockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
        
        //  OSType sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer);
        //  int doReverseChannels;
        //  if (kCVPixelFormatType_32ARGB == sourcePixelFormat) {
        //    doReverseChannels = 1;
        //  } else if (kCVPixelFormatType_32BGRA == sourcePixelFormat) {
        //    doReverseChannels = 0;
        //  } else {
        //    assert(false);  // Unknown source format
        //  }
        //
        //  const int sourceRowBytes = (int)CVPixelBufferGetBytesPerRow(pixelBuffer);
        //  const int image_width = (int)CVPixelBufferGetWidth(pixelBuffer);
        //  const int fullHeight = (int)CVPixelBufferGetHeight(pixelBuffer);
        //  CVPixelBufferLockBaseAddress(pixelBuffer, 0);
        //  unsigned char *sourceBaseAddr =
        //      (unsigned char *)(CVPixelBufferGetBaseAddress(pixelBuffer));
        //  int image_height;
        //  unsigned char *sourceStartAddr;
        //  if (fullHeight <= image_width) {
        //    image_height = fullHeight;
        //    sourceStartAddr = sourceBaseAddr;
        //  } else {
        //    image_height = image_width;
        //    const int marginY = ((fullHeight - image_width) / 2);
        //    sourceStartAddr = (sourceBaseAddr + (marginY * sourceRowBytes));
        //  }
        //  const int image_channels = 4;
        
        
        
        
//        LOG(INFO) << "tensor AFTER = "<< std::endl << " "  << image_tensor.flat<float>() << std::endl << std::endl;
        
        //    for (int y = 0; y < height; ++y) {
        //        const float* source_row = source_data + (y * width * depth);
        //        for (int x = 0; x < width; ++x) {
        //            const float* source_pixel = source_row + (x * depth);
        //            for (int c = 0; c < depth; ++c) {
        //                const float* source_value = source_pixel + c;
        //                input_tensor_mapped(0, y, x, c) = *source_value;
        //            }
        //        }
        //    }
        
        
    }
    
}

+ (std::vector<dlib::rectangle>)convertCGRectValueArray:(NSArray<NSValue *> *)rects {
    std::vector<dlib::rectangle> myConvertedRects;
    for (NSValue *rectValue in rects) {
        CGRect rect = [rectValue CGRectValue];
        long left = rect.origin.x;
        long top = rect.origin.y;
        long right = left + rect.size.width;
        long bottom = top + rect.size.height;
        dlib::rectangle dlibRect(left, top, right, bottom);
        
        myConvertedRects.push_back(dlibRect);
    }
    return myConvertedRects;
}

- (void)runCNNOnFrame:(cv::Mat)img {
    auto now = Clock::now();
    
    milliseconds diff = duration_cast<milliseconds>(now - lastRunCNN);
    std::cout << "Timing:"<< diff.count() << "ms" << std::endl;
    std::cout << "FPS:"<< 1000.0/diff.count() << std::endl;
    lastRunCNN = now;
    int h = storeImg.rows;
    int w = storeImg.cols;
    int d = storeImg.channels();
    
    int wanted_input_height = 128;
    int wanted_input_width = 128;
    int wanted_input_channels = 1;
    const float input_mean = 0.0f;
    const float input_std = 255.0f;
    tensorflow::Tensor image_tensor(tensorflow::DT_FLOAT,
                                    tensorflow::TensorShape({1, wanted_input_height, wanted_input_width, 1}));
    //tensor = image_tensor;
    auto input_tensor_mapped = image_tensor.tensor<float, 4>();
    
    UInt8* source_data = (tensorflow::uint8*)storeImg.data;
    Float32 *out = (Float32*)input_tensor_mapped.data();
    LOG(INFO) << "source data = "<< std::endl << " "  << *(float*)source_data << std::endl << std::endl;
    for (int y = 0; y < wanted_input_height; ++y) {
        Float32 *out_row = out + (y * wanted_input_width * wanted_input_channels);
        const UInt8* source_row = source_data + (y * w * d);
        for (int x = 0; x < wanted_input_width; ++x) {
            const UInt8* source_pixel = source_row + (x * d);
            Float32 *out_pixel = out_row + (x * wanted_input_channels);
            for (int c = 0; c < wanted_input_channels; ++c) {
                const UInt8* source_value = source_pixel + c;
                if (source_value != nullptr && out_pixel != nullptr) {
                    out_pixel[0] = (*source_value)/input_std;
                }
                
            }
        }
    }
    if (tf_session.get()) {
        std::vector<tensorflow::Tensor> outputs;
        LOG(INFO) << "Running model...\n";
        tensorflow::Status run_status = tf_session->Run(
                                                        {{input_layer_name, image_tensor}}, {output_layer_name}, {}, &outputs);
        if (!run_status.ok()) {
            LOG(ERROR) << "Running model failed:" << run_status;
        } else {
            tensorflow::Tensor *output = &outputs[0];
            auto f = output->flat<float>();
            //    float *array = (float *)malloc(sizeof(float) * 256);
            //    for (int i = 0; i < f.size(); i++) {
            //        array[i] = f.data()[i];
            //    }
            //
            
            auto predictions = f;
            float* new_arr = new float[256]();
            const int count = f.size();
            for (int i = 0; i < count; ++i) {
                const float value = f(i);
                new_arr[i] = value;
            }
            result = new_arr;
            
            if (lockFace != nil) {
                double cos = cosineSimilarity(new_arr,lockFace);
                std::cout << "Similarity:" << cos << std::endl;
                if (cos > 0.40) {
                    
                    if (countSuccess > 1) {
                        dispatch_async(dispatch_get_main_queue(), ^{
                            [A setText:[NSString stringWithFormat:@"%.20lf", cos]];
                            [A setTextColor:[UIColor greenColor]];
                        });
                        countSuccess--;
                    } else {
                        countSuccess++;
                    }
                    
                    
                    
                } else if (cos < 0.20) {
                    [self clearCounts];
                    dispatch_async(dispatch_get_main_queue(), ^{
                        [A setText:[NSString stringWithFormat:@"%.20lf", cos]];
                        [A setTextColor:[UIColor redColor]];
                    });
                    
                    
                } else {
                    dispatch_async(dispatch_get_main_queue(), ^{
                        [A setText:[NSString stringWithFormat:@"%.20lf", cos]];
                        [A setTextColor:[UIColor yellowColor]];
                    });
                }
                
            }
            LOG(INFO) << "Running model success: " << run_status;
            
            //                      NSMutableDictionary *newValues = [NSMutableDictionary dictionary];
            //                      for (int index = 0; index < predictions.size(); index += 1) {
            //                        const float predictionValue = predictions(index);
            //                        if (predictionValue > 0.05f) {
            //                          std::string label = labels[index % predictions.size()];
            //                          NSString *labelObject = [NSString stringWithCString:label.c_str()];
            //                          NSNumber *valueObject = [NSNumber numberWithFloat:predictionValue];
            //                          [newValues setObject:valueObject forKey:labelObject];
            //                        }
            //                      }
            //                dispatch_async(dispatch_get_main_queue(), ^(void) {
            //                            [self setPredictionValues:newValues];
            //                });
            
        }
    }

}

- (void)dealloc {
    [self teardownAVCapture];
    [A release];
    [super dealloc];
}

// use front/back camera
- (IBAction)switchCameras:(id)sender {
    AVCaptureDevicePosition desiredPosition;
    if (isUsingFrontFacingCamera)
        desiredPosition = AVCaptureDevicePositionBack;
    else
        desiredPosition = AVCaptureDevicePositionFront;
    
    for (AVCaptureDevice *d in
         [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo]) {
        if ([d position] == desiredPosition) {
            [[previewLayer session] beginConfiguration];
            AVCaptureDeviceInput *input =
            [AVCaptureDeviceInput deviceInputWithDevice:d error:nil];
            for (AVCaptureInput *oldInput in [[previewLayer session] inputs]) {
                [[previewLayer session] removeInput:oldInput];
            }
            [[previewLayer session] addInput:input];
            [[previewLayer session] commitConfiguration];
            break;
        }
    }
    isUsingFrontFacingCamera = !isUsingFrontFacingCamera;
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
}

- (void)handleTapGesture:(UITapGestureRecognizer *)sender {
    if (sender.state == UIGestureRecognizerStateRecognized) {
        [self switchCameras:nil];
    }
}

- (void)viewDidLoad {
    [super viewDidLoad];
    [self clearCounts];
    UITapGestureRecognizer *tapGesture = [[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(handleTapGesture:)];
    tapGesture.numberOfTapsRequired = 2;
    [self.view addGestureRecognizer:tapGesture];
    [tapGesture release];
    
    result = new float[256]();
    labelLayers = [[NSMutableArray alloc] init];
    oldPredictionValues = [[NSMutableDictionary alloc] init];
    // Load TensorFlow
    tensorflow::Status load_status;
    if (model_uses_memory_mapping) {
        load_status = LoadMemoryMappedModel(
                                            model_file_name, model_file_type, &tf_session, &tf_memmapped_env);
    } else {
        load_status = LoadModel(model_file_name, model_file_type, &tf_session);
    }
    if (!load_status.ok()) {
        LOG(FATAL) << "Couldn't load model: " << load_status;
    }
    
    tensorflow::Status labels_status =
    LoadLabels(labels_file_name, labels_file_type, &labels);
    if (!labels_status.ok()) {
        LOG(FATAL) << "Couldn't load labels: " << labels_status;
    }
    
    // Load dlib files
    if (dlib_on) {
        NSString *modelFileName = [[NSBundle mainBundle] pathForResource:@"shape_predictor_68_face_landmarks" ofType:@"dat"];
        std::string modelFileNameCString = [modelFileName UTF8String];
        
        dlib::deserialize(modelFileNameCString) >> sp;
    }
    
    
    
    NSDictionary *detectorOptions = [[NSDictionary alloc] initWithObjectsAndKeys:CIDetectorAccuracyLow, CIDetectorAccuracy, nil];
    self.faceDetector = [CIDetector detectorOfType:CIDetectorTypeFace context:nil options:detectorOptions];
    //    auto a = [self process:1];
    //        auto b =[self process:2];
    //
    //        LOG(INFO) << "cosSim" << cosineSimilarity(a,b);
    [self setupAVCapture];
}




- (void)viewDidUnload {
    [super viewDidUnload];
    [oldPredictionValues release];
}

- (void)viewWillAppear:(BOOL)animated {
    [super viewWillAppear:animated];
}

- (void)viewDidAppear:(BOOL)animated {
    [super viewDidAppear:animated];
}

- (void)viewWillDisappear:(BOOL)animated {
    [super viewWillDisappear:animated];
}

- (void)viewDidDisappear:(BOOL)animated {
    [super viewDidDisappear:animated];
}

- (BOOL)shouldAutorotateToInterfaceOrientation:
(UIInterfaceOrientation)interfaceOrientation {
    return (interfaceOrientation == UIInterfaceOrientationPortrait);
}

- (BOOL)prefersStatusBarHidden {
    return YES;
}



cv::Mat align2(cv::Mat &img, std::vector<dlib::point> &f3pt,
              int crop_size, int ec_mc_y, int ec_y) {
    //    std::cout << " NOW this is leye INSIDE " << f5pt[0].x() << " " << f5pt[0].y() << "\n";
    //cv::Mat img;
    //cv::cvtColor(image, img, cv::COLOR_BGRA2GRAY);
    
    double ang_tan = atan((1.0*f3pt[0].y()- f3pt[1].y())/(1.0*f3pt[0].x()- f3pt[1].x()));
    double ang = ang_tan/M_PI*180;
    cv::Mat img_rot;
//    if (ang < 0.05) {
//        ang = 0;
//        img_rot = img;
//    } else {
//        img_rot = rotate(img, ang);
//    }
    img_rot = rotate(img, ang);
    int imgh = img_rot.rows;
    int imgw = img_rot.cols;
    
    double x = (f3pt[0].x() + f3pt[1].x())/2.0;
    double y = (f3pt[0].y() + f3pt[1].y())/2.0;
    ang = -ang_tan;
    
    cv::Point2d eyec = transform(x,y, ang, img.rows, img.cols, img_rot.rows, img_rot.cols);
    
    double xn = f3pt[2].x();
    double yn = f3pt[2].y();
    
    cv::Point2d mouthc = transform(xn,yn, ang, img.rows, img.cols, img_rot.rows, img_rot.cols);
    
    double resize_scale = fabs(ec_mc_y/(mouthc.y-eyec.y));
    if (resize_scale > 1.0) {
        resize_scale = fabs(ec_mc_y/(yn-y));
    }
    resize_scale = fmin(resize_scale, 2.0);
    cv::Size resize_size(resize_scale*img_rot.cols,resize_scale*img_rot.rows);
    std::cout << "diff" << (yn - y) << std::endl;
    std::cout << "resize_scale" << resize_scale << std::endl;
    std::cout << "ang" << (ang) << std::endl;
    std::cout << "Eyes" << (eyec.y) << std::endl;
    cv::Mat img_resize;
    
    cv::resize(img_rot, img_resize, resize_size);
    
    double eyec2_x = round((eyec.x - imgw/2)*resize_scale+ img_resize.cols/2.0);
    double eyec2_y = round((eyec.y - imgh/2.0)*resize_scale+ img_resize.rows/2.0);
    
    int crop_y = fmax(eyec2_y - ec_y, 1);
    
    int crop_y_end = fmin(crop_y + crop_size, img_resize.rows-1);
    
    int crop_x = fmax((eyec2_x - crop_size/2),1);
    
    int crop_x_end = fmin(crop_x + crop_size, img_resize.cols-1);
    
    cv::Rect crop_roi(crop_x, crop_y, crop_x_end-crop_x, crop_y_end-crop_y);
    
    cv::Mat croppedImage = img_resize(crop_roi);
    
    cv::Mat im_gray;
    
    
    cv::Mat croppedImage8u;
    croppedImage.convertTo(croppedImage8u,CV_8U);
    cv::cvtColor(croppedImage8u,im_gray,CV_BGR2GRAY);
    
    
    return im_gray;
    
}



cv::Mat align(cv::Mat &img, std::vector<dlib::point> &f5pt,
              int crop_size, int ec_mc_y, int ec_y) {
    //    std::cout << " NOW this is leye INSIDE " << f5pt[0].x() << " " << f5pt[0].y() << "\n";
    //cv::Mat img;
    //cv::cvtColor(image, img, cv::COLOR_BGRA2GRAY);
    
    double ang_tan = atan((1.0*f5pt[0].y()- f5pt[1].y())/(1.0*f5pt[0].x()- f5pt[1].x()));
    double ang = ang_tan/M_PI*180;
    cv::Mat img_rot;
    if (ang < 0.10) {
        ang = 0;
        img_rot = img;
    } else {
        img_rot = rotate(img, ang);
    }
    
    int imgh = img_rot.rows;
    int imgw = img_rot.cols;
    
    double x = (f5pt[0].x() + f5pt[1].x())/2.0;
    double y = (f5pt[0].y() + f5pt[1].y())/2.0;
    ang = -ang_tan;
    
    cv::Point2d eyec = transform(x,y, ang, img.rows, img.cols, img_rot.rows, img_rot.cols);
    
    double xn = (f5pt[3].x() + f5pt[4].x())/2.0;
    double yn = (f5pt[3].y() + f5pt[4].y())/2.0;
    
    cv::Point2d mouthc = transform(xn,yn, ang, img.rows, img.cols, img_rot.rows, img_rot.cols);
    
    double resize_scale = fabs(ec_mc_y/(mouthc.y-eyec.y));
    if (resize_scale > 1.0) {
        resize_scale = fabs(ec_mc_y/(yn-y));
    }
    cv::Size resize_size(resize_scale*img_rot.cols,resize_scale*img_rot.rows);
    std::cout << "diff" << (yn - y) << std::endl;
    std::cout << "resize_scale" << resize_scale << std::endl;
    std::cout << "ang" << (ang) << std::endl;
    std::cout << "Eyes" << (eyec.y) << std::endl;
    cv::Mat img_resize;
    
    cv::resize(img_rot, img_resize, resize_size);
    
    double eyec2_x = round((eyec.x - imgw/2)*resize_scale+ img_resize.cols/2.0);
    double eyec2_y = round((eyec.y - imgh/2.0)*resize_scale+ img_resize.rows/2.0);
    
    int crop_y = fmax(eyec2_y - ec_y, 1);
    
    int crop_y_end = fmin(crop_y + crop_size, img_resize.rows);
    
    int crop_x = fmax((eyec2_x - crop_size/2),1);
    
    int crop_x_end = fmin(crop_x + crop_size, img_resize.cols);
    
    cv::Rect crop_roi(crop_x, crop_y, crop_x_end-crop_x, crop_y_end-crop_y);
    
    cv::Mat croppedImage = img_resize(crop_roi);
    
    cv::Mat im_gray;
 
    
    cv::Mat croppedImage8u;
    croppedImage.convertTo(croppedImage8u,CV_8U);
    cv::cvtColor(croppedImage8u,im_gray,CV_BGR2GRAY);
    
    char buffer[256];
    
    //HOME is the home directory of your application
    //points to the root of your sandbox
    strcpy(buffer,getenv("HOME"));
    
    //concatenating the path string returned from HOME
    strcat(buffer,"/Documents/my.txt");
    
    //Creates an empty file for writing
    
    
    std::ofstream fout(buffer);
    for(int i=0; i<im_gray.rows; i++){
        for(int j=0; j<im_gray.cols; j++){
            fout<<unsigned(im_gray.at<uint8>(i,j))<<"\t";
        }
        fout<<std::endl;
    }
    fout.close();
    
    
    return im_gray;
    
}



@end
