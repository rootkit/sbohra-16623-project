// Copyright 2016 Sudev Bohra All rights reserved.
//
// Created by Sudev Bohra on 12/10/16.
//
// Not for commercial use.

#include <metal_stdlib>
#include <metal_math>
using namespace metal;


kernel void adjust_mean_rgb(texture2d<float, access::read> inTexture [[texture(0)]],
                            texture2d<float, access::write> outTexture [[texture(1)]],
                            uint2 gid [[thread_position_in_grid]]) {
  float4 inColor = inTexture.read(gid);
  float4 outColor = float4(inColor.z*255.0 - 103.939, inColor.y*255.0 - 116.779, inColor.x*255.0 - 123.68, 0.0);
  outTexture.write(outColor, gid);
}

kernel void adjust_mean_bgr(texture2d<float, access::read> inTexture [[texture(0)]],
                            texture2d<float, access::write> outTexture [[texture(1)]],
                            uint2 gid [[thread_position_in_grid]]) {
  float4 inColor = inTexture.read(gid);
  float4 outColor = float4(inColor.x*255.0 - 103.939, inColor.y*255.0 - 116.779, inColor.z*255.0 - 123.68, 0.0);
  outTexture.write(outColor, gid);
}


// Rec 709 LUMA values for grayscale image conversion
constant float3 kRec709Luma = float3(0.2126, 0.7152, 0.0722);




// Grayscale compute shader
kernel void grayscale(texture2d<float, access::read>  inTexture   [[ texture(0) ]],
                      texture2d<float, access::write> outTexture  [[ texture(1) ]],
                      uint2                          gid         [[ thread_position_in_grid ]])
{
    if((gid.x < outTexture.get_width()) && (gid.y < outTexture.get_height()))
    {
        float4 inColor  = inTexture.read(gid);
        float  gray     = dot(inColor.rgb, kRec709Luma);
//        float gray = inColor.r;
//        float ten = 10;
//        float nine = 9;
//        gray = (floor((gray * pow(ten, nine) + 0.5)) / pow(ten, nine));
        float4 outColor = float4(gray.r, 0.0,0.0,0.0);
        
        outTexture.write(outColor, gid);
    }
}


// Max feature map compute shader
kernel void max_feature_map(texture2d_array<float, access::read>  inTexture   [[ texture(0) ]],
                      texture2d_array<float, access::write> outTexture  [[ texture(1) ]],
                      uint2                          gid         [[ thread_position_in_grid ]])
{
    
    if((gid.x < outTexture.get_width()) && (gid.y < outTexture.get_height()))
    {
        uint in_array_size = inTexture.get_array_size();
        uint out_array_size = outTexture.get_array_size();
        
        for (uint x = 0; x < out_array_size; x++) {
            float4 inxColor1  = inTexture.read(gid, x);
            float4 inxColor2  = inTexture.read(gid, x+out_array_size);
            
            float4 outxColor = float4(max(inxColor1[0],inxColor2[0]), max(inxColor1[1],inxColor2[1]), max(inxColor1[2],inxColor2[2]), max(inxColor1[3],inxColor2[3]));
            outTexture.write(outxColor, gid, x);
        }
        
    }
}





//
//kernel void max_feature_map(texture2d_array<float, access::read> inTexture [[texture(0)]],
//                            texture2d_array<float, access::write> outTexture [[texture(1)]],
//                            uint2 gid [[thread_position_in_grid]]) {
//    float4 inColor = inTexture.read(gid);
//    
//    float4 outColor = float4(inColor.x*255.0 - 103.939, inColor.y*255.0 - 116.779, inColor.z*255.0 - 123.68, 0.0);
//    outTexture.write(outColor, gid);
//}
