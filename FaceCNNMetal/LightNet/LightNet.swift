// Copyright 2016 Sudev Bohra All rights reserved.
//
// Created by Sudev Bohra on 12/10/16.
//
// Not for commercial use.

import MetalPerformanceShaders
import QuartzCore





private func makeConv(device: MTLDevice,
                      inDepth: Int,
                      outDepth: Int,
                      weights: UnsafePointer<Float>,
                      bias: UnsafePointer<Float>, kernelSize: Int = 1) -> MPSCNNConvolution {


  let relu : MPSCNNNeuronReLU? = nil

  let desc = MPSCNNConvolutionDescriptor(kernelWidth: kernelSize,
                                         kernelHeight: kernelSize,
                                         inputFeatureChannels: inDepth,
                                         outputFeatureChannels: outDepth,
                                         neuronFilter: relu)
  desc.strideInPixelsX = 1
  desc.strideInPixelsY = 1

  let conv = MPSCNNConvolution(device: device,
                               convolutionDescriptor: desc,
                               kernelWeights: weights,
                               biasTerms: bias,
                               flags: MPSCNNConvolutionFlags.none)


  conv.edgeMode = .zero

  return conv
}

private func makePool(device: MTLDevice) -> MPSCNNPoolingMax {

  return MPSCNNPoolingMax(device: device,
                          kernelWidth: 2,
                          kernelHeight: 2,
                          strideInPixelsX: 2,
                          strideInPixelsY: 2)
}

private func makeFC(device: MTLDevice,
                    inExtent: Int,
                    inDepth: Int,
                    fanOut: Int,
                    weights: UnsafePointer<Float>,
                    bias: UnsafePointer<Float>,
                    withRelu: Bool = false) -> MPSCNNFullyConnected {


  let filter: MPSCNNNeuron? = withRelu ? MPSCNNNeuronReLU(device: device, a: 0) : nil

  let desc = MPSCNNConvolutionDescriptor(kernelWidth: inExtent,
                                         kernelHeight: inExtent,
                                         inputFeatureChannels: inDepth,
                                         outputFeatureChannels: fanOut,
                                         neuronFilter: filter)

  let fc = MPSCNNFullyConnected(device: device,
                                convolutionDescriptor: desc,
                                kernelWeights: weights,
                                biasTerms: bias,
                                flags: MPSCNNConvolutionFlags.none)
  return fc
}

/*
  Implements the LightNet neural network.
 

*/
public class LightNet {
  let device: MTLDevice
  let commandQueue: MTLCommandQueue

  // The custom compute kernels for preprocessing the input images.
  let pipelineG: MTLComputePipelineState
  let pipelineMfm: MTLComputePipelineState


  let outputImage: MPSImage


  let lanczos: MPSImageLanczosScale



    
    let conv1 : MPSCNNConvolution
    let pool1  : MPSCNNPoolingMax
    
    let conv2a : MPSCNNConvolution
    let conv2 : MPSCNNConvolution
    let pool2  : MPSCNNPoolingMax
    
    let conv3a : MPSCNNConvolution
    let conv3 : MPSCNNConvolution
    let pool3  : MPSCNNPoolingMax
    
    let conv4a : MPSCNNConvolution
    let conv4 : MPSCNNConvolution
    
    let conv5a : MPSCNNConvolution
    let conv5 : MPSCNNConvolution
    let pool4  : MPSCNNPoolingMax
    
    let fc1: MPSCNNFullyConnected
    


    let input_id  = MPSImageDescriptor(channelFormat: .float16, width: 128, height: 128, featureChannels: 1)

    let conv1_id  = MPSImageDescriptor(channelFormat: .float16, width: 128, height: 128, featureChannels: 96)
    
    let mfm1_id  = MPSImageDescriptor(channelFormat: .float16, width: 128, height: 128, featureChannels: 48)

    let pool1_id  = MPSImageDescriptor(channelFormat: .float16, width: 64, height: 64, featureChannels: 48)

    let conv2a_id  = MPSImageDescriptor(channelFormat: .float16, width: 64, height: 64, featureChannels: 96)
    let mfm2a_id  = MPSImageDescriptor(channelFormat: .float16, width: 64, height: 64, featureChannels: 48)
    let conv2_id  = MPSImageDescriptor(channelFormat: .float16, width: 64, height: 64, featureChannels: 192)
    let mfm2_id  = MPSImageDescriptor(channelFormat: .float16, width: 64, height: 64, featureChannels: 96)


    let pool2_id  = MPSImageDescriptor(channelFormat: .float16, width:  32, height:  32, featureChannels: 96)
    
    let conv3a_id  = MPSImageDescriptor(channelFormat: .float16, width: 32, height: 32, featureChannels: 192)
    let mfm3a_id  = MPSImageDescriptor(channelFormat: .float16, width: 32, height: 32, featureChannels: 96)
    let conv3_id  = MPSImageDescriptor(channelFormat: .float16, width: 32, height: 32, featureChannels: 384)
    let mfm3_id  = MPSImageDescriptor(channelFormat: .float16, width: 32, height: 32, featureChannels: 192)
    
    
    let pool3_id  = MPSImageDescriptor(channelFormat: .float16, width:  16, height:  16, featureChannels: 192)
    
    let conv4a_id  = MPSImageDescriptor(channelFormat: .float16, width: 16, height: 16, featureChannels: 384)
    let mfm4a_id  = MPSImageDescriptor(channelFormat: .float16, width: 16, height: 16, featureChannels: 192)
    let conv4_id  = MPSImageDescriptor(channelFormat: .float16, width: 16, height: 16, featureChannels: 256)
    let mfm4_id  = MPSImageDescriptor(channelFormat: .float16, width: 16, height: 16, featureChannels: 128)
    
    
    let conv5a_id  = MPSImageDescriptor(channelFormat: .float16, width: 16, height: 16, featureChannels: 256)
    let mfm5a_id  = MPSImageDescriptor(channelFormat: .float16, width: 16, height: 16, featureChannels: 128)
    let conv5_id  = MPSImageDescriptor(channelFormat: .float16, width: 16, height: 16, featureChannels: 256)
    let mfm5_id  = MPSImageDescriptor(channelFormat: .float16, width: 16, height: 16, featureChannels: 128)
    
    
    let pool4_id  = MPSImageDescriptor(channelFormat: .float16, width:  8, height:  8, featureChannels: 128)

    let fc1_id     = MPSImageDescriptor(channelFormat: .float16, width:   1, height:   1, featureChannels: 512)
    
    let mfm_fc1_id     = MPSImageDescriptor(channelFormat: .float16, width:   1, height:   1, featureChannels: 256)
    
    let output_id     = MPSImageDescriptor(channelFormat: .float16, width: 1, height: 128, featureChannels: 96)


  public init(device: MTLDevice) {
    print("Setting up neural network...")
    let startTime = CACurrentMediaTime()

    self.device = device
    commandQueue = device.makeCommandQueue()

    outputImage = MPSImage(device: device, imageDescriptor: output_id)

    do {
      let library = device.newDefaultLibrary()!
      let max_feature_map = library.makeFunction(name: "max_feature_map")
      pipelineMfm = try device.makeComputePipelineState(function: max_feature_map!)
      let grayscale = library.makeFunction(name: "grayscale")
      pipelineG = try device.makeComputePipelineState(function: grayscale!)
      
    } catch {
      fatalError("Error initializing compute pipeline")
    }


    guard let path = Bundle.main.path(forResource: "parameters", ofType: "data"),
          let blob = LightNetData(path: path) else {
      fatalError("Error loading network parameters")
    }

    lanczos = MPSImageLanczosScale(device: device)

    conv1 = makeConv(device: device, inDepth:   1, outDepth:  96, weights: blob.conv1_w, bias: blob.conv1_b, kernelSize: 5)
    
    pool1   = makePool(device: device)

    conv2a = makeConv(device: device, inDepth: 48, outDepth: 96, weights: blob.conv2a_w, bias: blob.conv2a_b)
    
    conv2 = makeConv(device: device, inDepth:  48, outDepth: 192, weights: blob.conv2_w, bias: blob.conv2_b, kernelSize: 3)
    
    pool2   = makePool(device: device)
    

    conv3a = makeConv(device: device, inDepth: 96, outDepth: 192, weights: blob.conv3a_w, bias: blob.conv3a_b)
    conv3 = makeConv(device: device, inDepth: 96, outDepth: 384, weights: blob.conv3_w, bias: blob.conv3_b, kernelSize: 3)
    
    pool3   = makePool(device: device)

    conv4a = makeConv(device: device, inDepth: 192, outDepth: 384, weights: blob.conv4a_w, bias: blob.conv4a_b)
    conv4 = makeConv(device: device, inDepth: 192, outDepth: 256, weights: blob.conv4_w, bias: blob.conv4_b, kernelSize: 3)
    

    conv5a = makeConv(device: device, inDepth: 128, outDepth: 256, weights: blob.conv5a_w, bias: blob.conv5a_b)
    conv5 = makeConv(device: device, inDepth: 128, outDepth: 256, weights: blob.conv5_w, bias: blob.conv5_b, kernelSize: 3)

    pool4   = makePool(device: device)

    fc1 = makeFC(device: device, inExtent: 8, inDepth:  128, fanOut: 512, weights: blob.fc1_w, bias: blob.fc1_b)



    let endTime = CACurrentMediaTime()
    print("Elapsed time: \(endTime - startTime) sec")
  }
    
    public func mfm_encode(commandBuffer : MTLCommandBuffer, sourceImage: MPSTemporaryImage, destinationImage: MPSImage) {
        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder.setComputePipelineState(pipelineMfm)
        encoder.setTexture(sourceImage.texture, at: 0)
        encoder.setTexture(destinationImage.texture, at: 1)
        let threadsPerGroups = MTLSizeMake(8, 8, 1)
        let threadGroups = MTLSizeMake(max(destinationImage.texture.width / threadsPerGroups.width,1),
                                       max(destinationImage.texture.height / threadsPerGroups.height,1), 1)
        
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroups)
        encoder.endEncoding()
        sourceImage.readCount -= 1
    }


  public func predict(image inputImage: MPSImage, bgr: Bool) -> [Float] {
    let startTime = CACurrentMediaTime()

    autoreleasepool{
      let commandBuffer = commandQueue.makeCommandBuffer()

      // This lets us squeeze some extra speed out of Metal.
      MPSTemporaryImage.prefetchStorage(with: commandBuffer, imageDescriptorList: [
        input_id, conv1_id, pool1_id, conv2a_id, conv2_id, pool2_id, conv3a_id, conv3_id, pool3_id,
        conv4a_id, conv4_id, conv5a_id, conv5_id, pool4_id, fc1_id, output_id ])

      // Scale the input image to 224x224 pixels.
      let img1 = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: input_id)
        
        
      let encoder = commandBuffer.makeComputeCommandEncoder()
      encoder.setComputePipelineState(pipelineG)
      encoder.setTexture(inputImage.texture, at: 0)
      encoder.setTexture(img1.texture, at: 1)
      let threadsPerGroups = MTLSizeMake(8, 8, 1)
      let threadGroups = MTLSizeMake(img1.texture.width / threadsPerGroups.width,
                                       img1.texture.height / threadsPerGroups.height, 1)
      encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroups)
      encoder.endEncoding()
      
//      let img2 = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: input_id)
//        
//      lanczos.encode(commandBuffer: commandBuffer, sourceTexture: img1.texture, destinationTexture: img2.texture)
//

      // Adjust the RGB values of each pixel to be in the range -128...127
      // by subtracting the "mean pixel". If the input texture is RGB, this 
      // also swaps the R and B values because the model expects BGR pixels. 
      // As far as I can tell there is no MPS shader that can do these things,
      // so we use a custom compute kernel.
//      let encoder = commandBuffer.makeComputeCommandEncoder()
//      encoder.setComputePipelineState(bgr ? pipelineBGR : pipelineRGB)
//      encoder.setTexture(img1.texture, at: 0)
//      encoder.setTexture(img2.texture, at: 1)
        

//      let threadsPerGroups = MTLSizeMake(8, 8, 1)
//      let threadGroups = MTLSizeMake(img2.texture.width / threadsPerGroups.width,
//                                     img2.texture.height / threadsPerGroups.height, 1)
//      encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroups)
//      encoder.endEncoding()
      //img1.readCount -= 1    // see MPSTemporaryImage docs why this is needed

      // Now we take the output from our custom shader and pass it through the
      // layers of the neural network. For each layer we use a new "temporary"
      // MPSImage to hold the results.

      let conv1_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv1_id)
        
        
      conv1.encode(commandBuffer: commandBuffer, sourceImage: img1, destinationImage: conv1_img)
//
        let mfm1_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: mfm1_id)
        mfm_encode(commandBuffer: commandBuffer, sourceImage: conv1_img, destinationImage: mfm1_img)
//      let conv1_2_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv1_id)
//      conv1_2.encode(commandBuffer: commandBuffer, sourceImage: conv1_1_img, destinationImage: conv1_2_img)
//
      let pool1_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: pool1_id)
      pool1.encode(commandBuffer: commandBuffer, sourceImage: mfm1_img, destinationImage: pool1_img)
      
//
    // Conv 2
      let conv2a_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv2a_id)
      conv2a.encode(commandBuffer: commandBuffer, sourceImage: pool1_img, destinationImage: conv2a_img)
      
        
      let mfm2a_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: mfm2a_id)
      mfm_encode(commandBuffer: commandBuffer, sourceImage: conv2a_img, destinationImage: mfm2a_img)
//
      let conv2_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv2_id)
      conv2.encode(commandBuffer: commandBuffer, sourceImage: mfm2a_img, destinationImage: conv2_img)

      let mfm2_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: mfm2_id)
      mfm_encode(commandBuffer: commandBuffer, sourceImage: conv2_img, destinationImage: mfm2_img)
      
//
      let pool2_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: pool2_id)
      pool2.encode(commandBuffer: commandBuffer, sourceImage: mfm2_img, destinationImage: pool2_img)
        
        //
        // Conv 3
        let conv3a_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv3a_id)
        conv3a.encode(commandBuffer: commandBuffer, sourceImage: pool2_img, destinationImage: conv3a_img)
        
        
        let mfm3a_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: mfm3a_id)
        mfm_encode(commandBuffer: commandBuffer, sourceImage: conv3a_img, destinationImage: mfm3a_img)
        //
        let conv3_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv3_id)
        conv3.encode(commandBuffer: commandBuffer, sourceImage: mfm3a_img, destinationImage: conv3_img)
        
        let mfm3_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: mfm3_id)
        mfm_encode(commandBuffer: commandBuffer, sourceImage: conv3_img, destinationImage: mfm3_img)
        
        //
        let pool3_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: pool3_id)
        pool3.encode(commandBuffer: commandBuffer, sourceImage: mfm3_img, destinationImage: pool3_img)
        
        // Conv 4
        let conv4a_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv4a_id)
        conv4a.encode(commandBuffer: commandBuffer, sourceImage: pool3_img, destinationImage: conv4a_img)
        
        
        let mfm4a_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: mfm4a_id)
        mfm_encode(commandBuffer: commandBuffer, sourceImage: conv4a_img, destinationImage: mfm4a_img)
        //
        let conv4_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv4_id)
        conv4.encode(commandBuffer: commandBuffer, sourceImage: mfm4a_img, destinationImage: conv4_img)
        
        let mfm4_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: mfm4_id)
        mfm_encode(commandBuffer: commandBuffer, sourceImage: conv4_img, destinationImage: mfm4_img)
        
        // Conv 5
        let conv5a_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv5a_id)
        conv3a.encode(commandBuffer: commandBuffer, sourceImage: mfm4_img, destinationImage: conv5a_img)
        
        
        let mfm5a_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: mfm5a_id)
        mfm_encode(commandBuffer: commandBuffer, sourceImage: conv5a_img, destinationImage: mfm5a_img)
        //
        let conv5_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv5_id)
        conv5.encode(commandBuffer: commandBuffer, sourceImage: mfm5a_img, destinationImage: conv5_img)
        
        let mfm5_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: mfm5_id)
        mfm_encode(commandBuffer: commandBuffer, sourceImage: conv5_img, destinationImage: mfm5_img)
        
        //
        let pool4_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: pool4_id)
        pool4.encode(commandBuffer: commandBuffer, sourceImage: mfm5_img, destinationImage: pool4_img)
        
        
        let fc1_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: fc1_id)
        fc1.encode(commandBuffer: commandBuffer, sourceImage: pool4_img, destinationImage: fc1_img)
        
        mfm_encode(commandBuffer: commandBuffer, sourceImage: fc1_img, destinationImage: outputImage)
        

      commandBuffer.commit()
      commandBuffer.waitUntilCompleted()
    }

    
    let result = self.outputImage.toFloatArray()

    let endTime = CACurrentMediaTime()
    print("Elapsed time: \(endTime - startTime) sec")
    print(result)
    return result
  }
}
