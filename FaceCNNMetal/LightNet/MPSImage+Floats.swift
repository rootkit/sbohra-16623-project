// Copyright 2016 Sudev Bohra All rights reserved.
//
// Created by Sudev Bohra on 12/10/16.
//
// Not for commercial use.

import Accelerate
import MetalPerformanceShaders

extension MPSImage {

  public func toFloatArray() -> [Float] {

    let count = self.width * self.height * self.featureChannels
    var outputFloat16 = [UInt16](repeating: 0, count: count)

    let region = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                           size: MTLSize(width: self.width, height: self.height, depth: 1))

    let numSlices = (self.featureChannels + 3)/4
    print(numSlices)
    for i in 0..<numSlices {
      self.texture.getBytes(&(outputFloat16[self.width * self.height * 4 * i]),
                            bytesPerRow: self.width * 4 * MemoryLayout<UInt16>.size,
                            bytesPerImage: 0,
                            from: region,
                            mipmapLevel: 0,
                            slice: i)
    }

    var outputFloat32 = [Float](repeating: 0, count: count)
    var bufferFloat16 = vImage_Buffer(data: &outputFloat16, height: 1, width: UInt(count), rowBytes: count * 2)
    var bufferFloat32 = vImage_Buffer(data: &outputFloat32, height: 1, width: UInt(count), rowBytes: count * 4)

    if vImageConvert_Planar16FtoPlanarF(&bufferFloat16, &bufferFloat32, 0) != kvImageNoError {
      print("Error converting float16 to float32")
    }

    return outputFloat32
  }
    
    
}
