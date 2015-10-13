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
using Mpdn.Extensions.Framework.RenderChain;
using Mpdn.Extensions.RenderScripts.Mpdn.Resizer;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Example
    {
        public class Basic : RenderChain
        {
            protected override string ShaderPath
            {
                get { return "Examples"; }
            }

            protected override IFilter CreateFilter(IFilter sourceFilter)
            {
                // get MPDN to scale image to target size first
                sourceFilter += new Resizer { ResizerOption = ResizerOption.TargetSize100Percent };

                // apply our blue tint
                var blueTint = CompileShader("BlueTintSm3.hlsl")
                    .Configure(linearSampling: false, arguments: new[] {0.25f, 0.5f, 0.75f});
                return new ShaderFilter(blueTint, sourceFilter);
            }
        }

        public class BasicExample : RenderChainUi<Basic>
        {
            public override ExtensionUiDescriptor Descriptor
            {
                get
                {
                    return new ExtensionUiDescriptor
                    {
                        Name = "SM3.0 Blue Tint Example",
                        Description = "(Example) Applies a blue tint over the image using Shader Model 3.0",
                        Guid = new Guid("3682DAD5-067C-4537-B540-BE86A7C3527A"),
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
