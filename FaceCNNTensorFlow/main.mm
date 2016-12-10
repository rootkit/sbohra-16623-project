// Copyright 2016 Sudev Bohra All rights reserved.
//
// Created by Sudev Bohra on 12/10/16.
//
// Not for commercial use.

#import <UIKit/UIKit.h>

#import "FaceCNNAppDelegate.h"

int main(int argc, char *argv[]) {
  int retVal = 0;

  @autoreleasepool {
    retVal = UIApplicationMain(
        argc, argv, nil, NSStringFromClass([FaceCNNAppDelegate class]));
  }
  return retVal;
}
