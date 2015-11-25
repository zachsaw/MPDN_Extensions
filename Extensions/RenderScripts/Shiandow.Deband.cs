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

using System;
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.Framework.RenderChain;
using Mpdn.RenderScript;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Shiandow.Deband
    {
        public class Deband : RenderChain
        {
            public int maxbitdepth { get; set; }
            public float power { get; set; }
            public bool grain { get; set; }

            public Deband()
            {
                maxbitdepth = 8;
                power = 0.5f;
                grain = true;
            }

            protected override ITextureFilter CreateFilter(ITextureFilter input)
            {
                if (Renderer.InputFormat.IsRgb())
                    return input;

                int bits = Renderer.InputFormat.GetBitDepth();
                if (bits > maxbitdepth) return input;

                float[] consts = {
                    (1 << bits) - 1, 
                    power
                };

                var Deband = CompileShader("Deband.hlsl", macroDefinitions: !grain ? "SkipDithering=1" : "")
                    .Configure(arguments: consts, perTextureLinearSampling: new[] { true, false });
                /*var Subtract = CompileShader("Subtract.hlsl")
                    .Configure(perTextureLinearSampling: new[] { false, true }, format: TextureFormat.Float16);
                var SubtractLimited = CompileShader("SubtractLimited.hlsl")
                    .Configure(perTextureLinearSampling: new[] { false, true }, arguments: Consts);*/

                ITextureFilter yuv = input.ConvertToYuv();
                var inputsize = yuv.Output.Size;

                var deband = yuv;
                double factor = 2.0;// 0.5 * Math.Sqrt(5) + 0.5;

                int max = (int)Math.Floor(Math.Log(Math.Min(inputsize.Width, inputsize.Height) / 3.0, factor));
                for (int i = max; i >= 0; i--)
                {
                    double scale = Math.Pow(factor, i);
                    var size = new TextureSize((int)Math.Round(inputsize.Width / scale), (int)Math.Round(inputsize.Height / scale));
                    if (size.Width == 0 || size.Height == 0) continue;
                    if (i == 0) size = inputsize;

                    deband = new ShaderFilter(Deband.Configure(transform: s => size), yuv, deband);
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
                        Copyright = "Deband by Shiandow"
                    };
                }
            }
        }
    }
}
