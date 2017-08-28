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
using Mpdn.Extensions.Framework.RenderChain.TextureFilter;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Shiandow.Deband
    {
        public class DebandExperimental : Deband
        {
            public DebandExperimental()
            {
                MaxBitDepth = 8;
                Power = 0.5f;
                PreserveDetail = true;
            }

            protected override ITextureFilter CreateFilter(ITextureFilter input)
            {
                if (Renderer.InputFormat.IsRgb())
                    return input;

                int bits = Renderer.InputFormat.GetBitDepth();
                if (bits > MaxBitDepth) return input;

                var composition = input as ICompositionFilter;
                var offset = new SharpDX.Vector2(0.0f, 0.0f); // TODO: process chroma offset properly

                float[] consts = {
                    (1 << bits) - 1,
                    Power
                };

                var Downscale = new Shader(FromFile("Downscale.hlsl"))
                {
                    Transform = s => new TextureSize(s.Width / 2, s.Height / 2),
                    Arguments = consts
                };

                var DownscaleLuma = new Shader(FromFile("DownscaleLuma.hlsl"))
                {
                    SizeIndex = 1,
                    Arguments = consts
                };
                DownscaleLuma["iteration"] = 0;

                var Deband = new Shader(FromFile("Deband.hlsl", macroDefinitions: PreserveDetail ? "PRESERVE_DETAIL=1" : ""))
                {
                    Arguments = consts,
                    LinearSampling = false,
                    SizeIndex = 1
                };

                ITextureFilter deband, chroma = null, luma = null;
                var pyramid = new Stack<ITextureFilter>();

                if (composition != null)
                {
                    deband = composition.Luma;
                    pyramid.Push(deband);
                    pyramid.Push(DownscaleLuma.ApplyTo(composition.Luma, composition.Chroma));
                }
                else
                {
                    deband = input.ConvertToYuv();
                    pyramid.Push(deband);
                }

                // Build gaussian pyramid
                while (pyramid.Peek().Size().Width >= 2
                    && pyramid.Peek().Size().Height >= 2)
                {
                    Downscale["iteration"] = pyramid.Count - 1;
                    pyramid.Push(Downscale.ApplyTo(pyramid.Peek()));
                }

                // Process pyramid
                var result = pyramid.Peek();
                while (pyramid.Count > 1)
                {
                    result = Deband.ApplyTo(pyramid.Pop(), pyramid.Peek(), result);
                    if (composition != null && pyramid.Count == 2)
                    {
                        chroma = result;
                        Deband.Format = TextureFormat.Unorm16_R;
                    }
                }

                if (composition != null)
                {
                    luma = result;
                    return new CompositionFilter(luma, chroma, composition.TargetSize, composition.ChromaOffset);
                }
                else return result.ConvertToRgb();
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
