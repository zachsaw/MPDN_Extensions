// This file is a part of MPDN Extensions.
// https://github.com/zachsaw/MPDN_Extensions
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
// 
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public
// License along with this library.
// 
using System;
using Mpdn.RenderScript.Mpdn.ImageProcessor;
using Mpdn.RenderScript.Mpdn.Resizer;
using Mpdn.RenderScript.Shiandow.Chroma;
using Mpdn.RenderScript.Shiandow.Nedi;

namespace Mpdn.RenderScript
{
    namespace MyOwnUniqueNameSpace // e.g. Replace with your user name
    {
        public class MyRenderChain : RenderChain
        {
            private string[] Deinterlace = { @"MPC-HC\Deinterlace (blend).hlsl" };
            private string[] PostProcess = { @"SweetFX\LumaSharpen.hlsl" };
            private string[] PreProcess = { @"SweetFX\Bloom.hlsl", @"SweetFX\LiftGammaGain.hlsl" };
            private string[] ToGamma = { @"ConvertToGammaLight.hlsl" };
            private string[] ToLinear = { @"ConvertToLinearLight.hlsl" };

            public override IFilter CreateFilter(IResizeableFilter sourceFilter)
            {
                // Scale chroma first (this bypasses MPDN's chroma scaler)
                sourceFilter += new BicubicChroma { Preset = Presets.MitchellNetravali };

                if (Renderer.InterlaceFlags.HasFlag(InterlaceFlags.IsInterlaced))
                {
                    // Deinterlace using blend
                    sourceFilter += new ImageProcessor { ShaderFileNames = Deinterlace };
                }

                // Pre resize shaders, followed by NEDI image doubler
                sourceFilter += new ImageProcessor { ShaderFileNames = PreProcess };

                // Use NEDI once only.
                // Note: To use NEDI as many times as required to get the image past target size,
                //       Change the following *if* to *while*
                if (IsUpscalingFrom(sourceFilter)) // See CombinedChain for other comparer methods
                {
                    sourceFilter += new Nedi { AlwaysDoubleImage = true };
                }

                if (IsDownscalingFrom(sourceFilter))
                {
                    // Use linear light for downscaling
                    sourceFilter += new ImageProcessor { ShaderFileNames = ToLinear }
                                  + new Resizer { ResizerOption = ResizerOption.TargetSize100Percent }
                                  + new ImageProcessor { ShaderFileNames = ToGamma };
                }
                else
                {
                    // Otherwise, scale with gamma light
                    sourceFilter += new Resizer { ResizerOption = ResizerOption.TargetSize100Percent };
                }

                if (Renderer.VideoSize.Width < 1920)
                {
                    // Sharpen only if video isn't full HD
                    sourceFilter += new ImageProcessor { ShaderFileNames = PostProcess };
                }

                return sourceFilter;
            }
        }

        public class MyRenderScript : RenderChainUi<MyRenderChain>
        {
            public override ExtensionUiDescriptor Descriptor
            {
                get
                {
                    return new ExtensionUiDescriptor
                    {
                        Name = "Custom Render Script Chain",
                        Description = "A customized render script chain (Advanced)",
                        Guid = new Guid("B0AD7BE7-A86D-4BE4-A750-4362FEF28A55"),
                        Copyright = "" // Optional field
                    };
                }
            }
        }
    }
}
