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
using Mpdn.Extensions.Framework;
using Mpdn.RenderScript;
using Mpdn.RenderScript.Scaler;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Shiandow.Deband
    {
        public class Deband : RenderChain
        {
            public int maxbitdepth { get; set; }
            public float margin { get; set; }
            public float power { get; set; }

            public Deband()
            {
                maxbitdepth = 8;
                margin = 0.0f;
                power = 0.5f;
            }

            public override IFilter CreateFilter(IFilter input)
            {
                var bilinear = new HwBilinear();
             
                int bits = 8;
                switch (Renderer.InputFormat)
                {
                    case FrameBufferInputFormat.P010: bits = 10; break;
                    case FrameBufferInputFormat.Y410: bits = 10; break;
                    case FrameBufferInputFormat.P016: bits = 16; break;
                    case FrameBufferInputFormat.Y416: bits = 16; break;
                    case FrameBufferInputFormat.Rgb24: return input;
                    case FrameBufferInputFormat.Rgb32: return input;
                }
                if (bits > maxbitdepth) return input;

                float[] YuvConsts = new float[2];
                switch (Renderer.Colorimetric)
                {
                    case YuvColorimetric.Auto: break;
                    case YuvColorimetric.FullRangePc601: YuvConsts = new[] { 0.114f, 0.299f, 0.0f }; break;
                    case YuvColorimetric.FullRangePc709: YuvConsts = new[] { 0.0722f, 0.2126f, 0.0f }; break;
                    case YuvColorimetric.FullRangePc2020: YuvConsts = new[] { 0.0593f, 0.2627f, 0.0f }; break;
                    case YuvColorimetric.ItuBt601: YuvConsts = new[] { 0.114f, 0.299f, 1.0f }; break;
                    case YuvColorimetric.ItuBt709: YuvConsts = new[] { 0.0722f, 0.2126f, 1.0f }; break;
                    case YuvColorimetric.ItuBt2020: YuvConsts = new[] { 0.0593f, 0.2627f, 1.0f }; break;
                }

                float[] Consts = new[] {
                    (1 << bits) - 1, 
                    power,
                    margin                    
                };

                var Deband = CompileShader("Deband.hlsl")
                    .Configure(arguments: Consts, perTextureLinearSampling: new[] { true, false });
                /*var Subtract = CompileShader("Subtract.hlsl")
                    .Configure(perTextureLinearSampling: new[] { false, true }, format: TextureFormat.Float16);
                var SubtractLimited = CompileShader("SubtractLimited.hlsl")
                    .Configure(perTextureLinearSampling: new[] { false, true }, arguments: Consts);*/

                IFilter yuv = input.ConvertToYuv();
                var inputsize = yuv.OutputSize;

                var deband = yuv;
                double phi = 0.5 * Math.Sqrt(5) + 0.5; // Use irrational factor to prevent blocking.
                for (int i = 8; i >= 0; i--)
                {
                    double factor = Math.Pow(phi, i);
                    var size = new TextureSize((int)Math.Round(inputsize.Width / factor), (int)Math.Round(inputsize.Height / factor));
                    if (size.Width == 0 || size.Height == 0) continue;
                    if (i == 0) size = inputsize;

                    deband = new ShaderFilter( Deband.Configure(transform: s => size), yuv, deband);
                }

                return deband.ConvertToRgb();
            }
        }

        public class DebandUi : RenderChainUi<Deband, DebandConfigDialog>
        {
            protected override string ConfigFileName
            {
                get { return "Shiandow.Deband"; }
            }

            public override string Category
            {
                get { return "Processing"; }
            }

            public override ExtensionUiDescriptor Descriptor
            {
                get
                {
                    return new ExtensionUiDescriptor
                    {
                        Guid = new Guid("EE3B46F7-00BB-4299-9B3F-058BCC3F591C"),
                        Name = "Deband",
                        Description = "Removes banding",
                        Copyright = "Deband by Shiandow",
                    };
                }
            }
        }
    }
}
