// Copyright 2016 Sudev Bohra All rights reserved.
//
// Created by Sudev Bohra on 12/10/16.
//
// Not for commercial use.

#import "FaceCNNAppDelegate.h"

@implementation FaceCNNAppDelegate

@synthesize window = _window;

- (BOOL)application:(UIApplication *)application
    didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
  [self.window makeKeyAndVisible];
  return YES;
}

- (void)applicationWillResignActive:(UIApplication *)application {
  [[UIApplication sharedApplication] setIdleTimerDisabled:NO];
}

- (void)applicationDidEnterBackground:(UIApplication *)application {
}

- (void)applicationWillEnterForeground:(UIApplication *)application {
}

- (void)applicationDidBecomeActive:(UIApplication *)application {
  [[UIApplication sharedApplication] setIdleTimerDisabled:YES];
}

- (void)applicationWillTerminate:(UIApplication *)application {
}

@end
