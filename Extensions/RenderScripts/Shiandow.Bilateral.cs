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
using SharpDX;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Shiandow.Bilateral
    {
        public class Bilateral : RenderChain, IChromaScaler
        {
            protected override ITextureFilter CreateFilter(ITextureFilter input)
            {
                return this.MakeChromaFilter(input);
            }

            public ITextureFilter CreateChromaFilter(ITextureFilter lumaInput, ITextureFilter chromaInput, TextureSize targetSize, Vector2 chromaOffset)
            {
                float[] yuvConsts = Renderer.Colorimetric.GetYuvConsts();

                var crossBilateral = CompileShader("CrossBilateral.hlsl")
                    .Configure(
                        arguments: new[] { chromaOffset.X, chromaOffset.Y, yuvConsts[0], yuvConsts[1] },
                        perTextureLinearSampling: new[] { true, false }
                    );

                // Fall back to default when downscaling is needed
                var chromaSize = chromaInput.Output.Size;
                if (targetSize.Width < chromaSize.Width || targetSize.Height < chromaSize.Height)
                    return null;

                var resizedLuma = lumaInput.SetSize(targetSize, tagged: true);

                return new ShaderFilter(crossBilateral, resizedLuma, chromaInput).ConvertToRgb();
            }
        }
        
        public class BilateralUi : RenderChainUi<Bilateral>
        {
            protected override string ConfigFileName
            {
                get { return "Bilateral"; }
            }

            public override string Category
            {
                get { return "Chroma Scaling"; }
            }

            public override ExtensionUiDescriptor Descriptor
            {
                get
                {
                    return new ExtensionUiDescriptor
                    {
                        Guid = new Guid("53534BBF-4749-4599-98C0-603302772B44"),
                        Name = "Bilateral",
                        Description = "Uses luma information to scale chroma",
                        Copyright = "Made by Shiandow",
                    };
                }
            }
        }
    }
}
