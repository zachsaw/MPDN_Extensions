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
using Mpdn.RenderScript;
using Mpdn.RenderScript.Mpdn.Resizer;

namespace Mpdn.RenderScripts
{
    namespace Example
    {
        public class DirectCompute : RenderChain
        {
            protected override string ShaderPath
            {
                get { return "Examples"; }
            }

            public override IFilter CreateFilter(IFilter sourceFilter)
            {
                if (!Renderer.IsDx11Avail)
                    return new NullFilter(); // display blank screen on purpose

                // get MPDN to scale image to target size first
                sourceFilter += new Resizer { ResizerOption = ResizerOption.TargetSize100Percent };

                // apply our blue tint
                var blueTint =
                    CompileShader11("BlueTintDirectCompute.hlsl", "cs_5_0")
                        .Configure(arguments: new[] {0.25f, 0.5f, 0.75f});
                var width = sourceFilter.OutputSize.Width;
                var height = sourceFilter.OutputSize.Height;
                return new DirectComputeFilter(blueTint, width/32 + 1, height/32 + 1, 1, sourceFilter);
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
