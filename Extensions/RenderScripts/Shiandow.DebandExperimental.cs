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
using System.Collections.Generic;
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.Framework.RenderChain;
using Mpdn.Extensions.Framework.RenderChain.Shader;
using Mpdn.RenderScript;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Shiandow.Deband
    {
        public class DebandExperimental : Deband
        {
            public DebandExperimental()
            {
                Power = 0.25f;
            }

            protected override ITextureFilter CreateFilter(ITextureFilter input)
            {
                if (Renderer.InputFormat.IsRgb())
                    return input;

                int bits = Renderer.InputFormat.GetBitDepth();
                if (bits > MaxBitDepth) return input;

                float[] consts = {
                    (1 << bits) - 1,
                    Power
                };

                var Downscale = new Shader(FromFile("Downscale.hlsl"))
                {
                    Transform = s => new TextureSize(s.Width/2, s.Height/2)
                };

                var Deband = new Shader(FromFile("Deband.hlsl", macroDefinitions: PreserveDetail ? "PRESERVE_DETAIL=1" : ""))
                {
                    Arguments = consts,
                    PerTextureLinearSampling = new[] {true, false, true},
                    SizeIndex = 1
                };

                var deband = input.ConvertToYuv();
                var pyramid = new Stack<ITextureFilter>();

                // Build gaussian pyramid
                pyramid.Push(deband);
                while (pyramid.Peek().Output.Size.Width  >= 2
                    && pyramid.Peek().Output.Size.Height >= 2)
                    pyramid.Push(Downscale.ApplyTo(pyramid.Peek()));

                var result = pyramid.Peek();
                while (pyramid.Count > 1)
                    result = Deband.ApplyTo(pyramid.Pop(), pyramid.Peek(), result);

                return result.ConvertToRgb();
            }
        }

        public class DebandExperimentalUi : RenderChainUi<DebandExperimental, DebandConfigDialog>
        {
            protected override string ConfigFileName
            {
                get { return "Shiandow.DebandExperimental"; }
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
                        Guid = new Guid("e386c8a2-e380-418e-a4a6-cb79d9f8c020"),
                        Name = "Deband Experimental",
                        Description = "Removes banding",
                        Copyright = "Deband Experimental by Shiandow"
                    };
                }
            }
        }
    }
}
