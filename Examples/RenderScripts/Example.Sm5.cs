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
using Mpdn.Extensions.Framework.RenderChain;
using Mpdn.Extensions.Framework.RenderChain.Filters;
using Mpdn.Extensions.RenderScripts.Mpdn.Resizer;
using Mpdn.RenderScript;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Example
    {
        public class Sm5 : RenderChain
        {
            protected override string ShaderPath
            {
                get { return "Examples"; }
            }

            protected override ITextureFilter CreateFilter(ITextureFilter input)
            {
                if (!Renderer.IsDx11Avail)
                    return new NullFilter(); // display blank screen on purpose

                // get MPDN to scale image to target size first
                input += new Resizer { ResizerOption = ResizerOption.TargetSize100Percent };

                // apply our blue tint
                var blueTint = new Shader11(FromFile("BlueTintSm5.hlsl", "ps_5_0"))
                {
                    LinearSampling = false,
                    Arguments = new[] { 0.25f, 0.5f, 0.75f }
                };

                return input.Apply(blueTint);
            }
        }

        public class Sm5Example : RenderChainUi<Sm5>
        {
            public override ExtensionUiDescriptor Descriptor
            {
                get
                {
                    return new ExtensionUiDescriptor
                    {
                        Name = "SM5.0 Blue Tint Example",
                        Description = "(Example) Applies a blue tint over the image using Shader Model 5.0",
                        Guid = new Guid("72594680-274B-464B-9BD0-6D99173B4555"),
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
