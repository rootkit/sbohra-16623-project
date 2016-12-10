// Copyright 2016 Sudev Bohra All rights reserved.
//
// Created by Sudev Bohra on 12/10/16.
//
// Not for commercial use.

import UIKit
import MetalKit
import MetalPerformanceShaders

class ViewController: UIViewController, VideoCaptureDelegate {

  @IBOutlet weak var spinner: UIActivityIndicatorView!
  @IBOutlet weak var spinnerPanel: UIView!

  @IBOutlet weak var mainPanel: UIView!
  @IBOutlet weak var cameraView: UIView!
  @IBOutlet weak var button: UIButton!
  @IBOutlet weak var imageView: UIImageView!
  @IBOutlet weak var predictionLabel: UILabel!

  private var device: MTLDevice!
  private var videoCapture: VideoCapture!
  private var textureLoader: MTKTextureLoader!
  private var nn: LightNet!

  override func viewDidLoad() {
    super.viewDidLoad()

    device = MTLCreateSystemDefaultDevice()

    // We use MetalKit's texture loader to create MPSImage objects.
    textureLoader = MTKTextureLoader(device: device)

    // Set up the live video feed.
    videoCapture = VideoCapture(device: device)
    videoCapture.delegate = self

    if let previewLayer = videoCapture.previewLayer {
      cameraView.layer.addSublayer(previewLayer)
    }
    //videoCapture.start()

    // Set up the UI.
    button.layer.cornerRadius = 10
    button.layer.borderColor = UIColor(white: 1, alpha: 0.1).cgColor
    button.layer.borderWidth = 1
    predictionLabel.text = ""

    // Show a message while the neural network is initializing.
    mainPanel.isHidden = true
    spinnerPanel.isHidden = false
    spinner.startAnimating()

    createNeuralNetwork {
      // Reveal the main UI for the app.
      self.spinner.stopAnimating()
      self.spinnerPanel.isHidden = true
      self.mainPanel.isHidden = false

      // Run the test image through the network. It should find a cat. ;)
       let start = Date();
      let v1 = self.predict(imageNamed: "Bill_Gates/Bill_Gates_0001.jpg")
        let v2 = self.predict(imageNamed: "Bill_Gates/Bill_Gates_0002.jpg")

        let end = Date();   // <<<<<<<<<<   end time
        
        
        let timeInterval: Double = end.timeIntervalSince(start); // <<<<< Difference in seconds (double)
        
        print("Time to evaluate problem: \(timeInterval) seconds");
      //let v2 = self.predict(imageNamed: "Hillary_Clinton/Hillary_Clinton_0002.jpg")
      //print(v1?.count)
      //print(self.cosineDistance(u: v1!, v: v2!))
    }
  }

  // MARK: - Neural network
 
    public func cosineDistance(u : [Float],v : [Float]) -> Float {
        let u2 = sqrt(u.map { $0*$0 }.reduce(0, +))
        let v2 = sqrt(v.map { $0*$0 }.reduce(0, +))
        let d = u2*v2
        var uu = u
        for i in 0..<256 {
            uu[i] = u[i]*v[i]
        }
        return uu.reduce(0,+)/d
    }
  private func createNeuralNetwork(completion: @escaping () -> Void) {
    // Make sure the current device supports MetalPerformanceShaders.
    guard MPSSupportsMTLDevice(device) else {
      print("Error: Metal Performance Shaders not supported on this device")
      return
    }

    // Because it takes a few seconds to load LightNet's parameters, perform
    // the construction of the neural network in a background thread.
    DispatchQueue.global().async {
      self.nn = LightNet(device: self.device)

      DispatchQueue.main.async(execute: completion)
    }
  }

  private func predict(imageNamed filename: String) -> [Float]! {
    print("Predicting \(filename)")

    if let texture = self.loadTexture(named: filename),
       let previewImage = UIImage(named: filename) {
      return predict(texture: texture, previewImage: previewImage, bgr: false)
    }
    return nil
  }

  private func predict(texture: MTLTexture, previewImage: UIImage, bgr: Bool) -> [Float]! {
    // Show a preview of the image.
    imageView.image = previewImage
    button.isEnabled = false
    let inputImage = self.image(from: texture)
    
    let prediction = self.nn.predict(image: inputImage, bgr: bgr)
    return prediction
//    DispatchQueue.global().async {
//      let inputImage = self.image(from: texture)
//      let prediction = self.nn.predict(image: inputImage, bgr: bgr)
//
//      DispatchQueue.main.async {
//        self.button.isEnabled = true
////        self.show(prediction: prediction)
//      }
//    }
  }

  // MARK: - Loading textures

  private func loadTexture(named filename: String) -> MTLTexture? {
    if let url = Bundle.main.url(forResource: filename, withExtension: "") {
      return loadTexture(url: url)
    } else {
      print("Error: could not find image \(filename)")
      return nil
    }
  }

  private func loadTexture(url: URL) -> MTLTexture? {
    do {
      // Note: the SRGB option should be set to false, otherwise the image
      // appears way too dark, since it wasn't actually saved as SRGB.
      return try textureLoader.newTexture(withContentsOf: url, options: [
        MTKTextureLoaderOptionSRGB : NSNumber(value: false)
      ])
    } catch {
      print("Error: could not load texture \(error)")
      return nil
    }
  }

  private func image(from texture: MTLTexture) -> MPSImage {
    // We set featureChannels to 3 because the neural network is only trained
    // on RGB data (the first 3 channels), not alpha (the 4th channel).
    
    return MPSImage(texture: texture, featureChannels: 1)
    
  }

  // MARK: - Video capture

  @IBAction func buttonTapped(_ sender: UIButton) {
    videoCapture.captureFrame()
  }

  func didCapture(texture: MTLTexture?, previewImage: UIImage?) {
    if let texture = texture, let previewImage = previewImage {
      predict(texture: texture, previewImage: previewImage, bgr: true)
    } else {
      imageView.image = nil
    }
  }

  // MARK: - UI stuff

  override func viewWillLayoutSubviews() {
    super.viewWillLayoutSubviews()
    videoCapture.previewLayer?.frame = cameraView.bounds
  }

  override func didReceiveMemoryWarning() {
    super.didReceiveMemoryWarning()
    print(#function)
  }

  override var preferredStatusBarStyle: UIStatusBarStyle {
    return .lightContent
  }
}
