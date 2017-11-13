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
using Mpdn.Extensions.Framework.RenderChain.TextureFilter;
using Mpdn.Extensions.RenderScripts.Mpdn.Resizer;
using Mpdn.RenderScript;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Example
    {
        public class DirectCompute : RenderChain
        {
            protected override string ShaderPath
            {
                get { return "Examples"; }
            }

            protected override ITextureFilter CreateFilter(ITextureFilter sourceFilter)
            {
                if (!Renderer.IsDx11Avail || Renderer.RenderQuality.PerformanceMode())
                    return new NullFilter(); // display blank screen on purpose

                // get MPDN to scale image to target size first
                sourceFilter += new Resizer { ResizerOption = ResizerOption.TargetSize100Percent };

                // apply our blue tint
                var width = sourceFilter.Size().Width;
                var height = sourceFilter.Size().Height;
                var blueTint =
                    new ComputeShader(FromFile("BlueTintDirectCompute.hlsl", "cs_5_0"))
                    {
                        ThreadGroupX = width / 32 + 1,
                        ThreadGroupY = height / 32 + 1,
                        Arguments = new[] { 0.25f, 0.5f, 0.75f }
                    };
                return sourceFilter.Apply(blueTint);
            }
        }

        public class DirectComputeExample : RenderChainUi<DirectCompute>
        {
            public override ExtensionUiDescriptor Descriptor
            {
                get
                {
                    return new ExtensionUiDescriptor
                    {
                        Name = "DirectCompute Blue Tint Example",
                        Description = "(Example) Applies a blue tint over the image using DirectCompute",
                        Guid = new Guid("2BAD9125-6474-42D4-9C65-9A03DE3280AF"),
                        Copyright = "" // Optional field
                    };
                }
            }

            public override string Category
            {
                get { return "Example"; }
            }
        }
    }
}
