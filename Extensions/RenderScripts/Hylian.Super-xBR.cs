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
using Mpdn.RenderScript;
using SharpDX;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Hylian.SuperXbr
    {
        public class SuperXbr : RenderChain
        {
            #region Settings

            public float EdgeStrength { get; set; }
            public float Sharpness { get; set; }
            public bool FastMethod { get; set; }
            public bool ThirdPass  { get; set; }

            public SuperXbr()
            {
                EdgeStrength = 1.0f;
                Sharpness = 1.0f;
                FastMethod = false;
                ThirdPass = true;
            }

            #endregion

            public override IFilter CreateFilter(IFilter input)
            {
                IFilter xbr;

                Func<TextureSize, TextureSize> transformWidth = s => new TextureSize(2 * s.Width, s.Height);
                Func<TextureSize, TextureSize> transformHeight = s => new TextureSize(s.Width, 2 * s.Height);
                Func<TextureSize, TextureSize> transform = s => new TextureSize(2 * s.Width, 2 * s.Height);

                float[] arguments = { EdgeStrength, Sharpness };
                string fastToggle = FastMethod ? "FAST_METHOD=1;" : "";

                var Pass0 = CompileShader("super-xbr.hlsl", entryPoint: "main_fragment", macroDefinitions: "Pass = 0;" + fastToggle).Configure(transform: transform, arguments: arguments);
                var Pass1 = CompileShader("super-xbr.hlsl", entryPoint: "main_fragment", macroDefinitions: "Pass = 1;" + fastToggle).Configure(arguments: arguments);
                var Pass2 = CompileShader("super-xbr.hlsl", entryPoint: "main_fragment", macroDefinitions: "Pass = 2;" + fastToggle).Configure(arguments: arguments);

                // Skip if downscaling
                if (Renderer.TargetSize.Width  <= input.OutputSize.Width 
                 && Renderer.TargetSize.Height <= input.OutputSize.Height)
                    return input;
                
                xbr = new ShaderFilter(Pass0, input);
                xbr = new ShaderFilter(Pass1, xbr);

                if (ThirdPass)
                    return xbr + (RenderChain)Pass2;
                else
                    return new ResizeFilter(xbr, xbr.OutputSize, new Vector2(0.5f, 0.5f));
            }

            protected override string ShaderPath
            {
                get { return "Super-xBR"; }
            }
        }

        public class SuperXbrUi : RenderChainUi<SuperXbr, SuperXbrConfigDialog>
        {
            protected override string ConfigFileName
            {
                get { return "Hylian.SuperXbr"; }
            }

            public override string Category
            {
                get { return "Upscaling"; }
            }

            public override ExtensionUiDescriptor Descriptor
            {
                get
                {
                    return new ExtensionUiDescriptor
                    {
                        Guid = new Guid("C19ED5B0-FC99-459A-B7BD-199BFDEA1F3C"),
                        Name = "Super-xBR",
                        Description = "Super-xBR image scaling",
                        Copyright = "Created by Hylian, ported to MPDN by Shiandow"
                    };
                }
            }
        }
    }
}
