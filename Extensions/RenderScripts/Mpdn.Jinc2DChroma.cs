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
using System.Drawing;
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.Framework.RenderChain;
using Mpdn.RenderScript;
using TransformFunc = System.Func<System.Drawing.Size, System.Drawing.Size>;
using WeightFilter = Mpdn.Extensions.Framework.RenderChain.TextureSourceFilter<Mpdn.ISourceTexture>;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Mpdn.Jinc2D
    {
        public class Jinc2DChroma : Jinc2D
        {
            protected override string ShaderPath
            {
                get { return "Jinc2D"; }
            }

            public override IFilter CreateFilter(IFilter input)
            {
                DiscardTextures();

                var sourceSize = input.OutputSize;
                if (!IsUpscalingFrom(sourceSize))
                    return input;

                var targetSize = Renderer.LumaSize;
                CreateWeights((Size) sourceSize, targetSize);

                var offset = Renderer.ChromaOffset;
                int lobes = TapCount.ToInt()/2;
                var shader = CompileShader("Jinc2D.hlsl",
                    macroDefinitions:
                        string.Format("LOBES = {0}; AR = {1}; CHROMA = 1;",
                            lobes, AntiRingingEnabled ? 1 : 0))
                    .Configure(
                        transform: size => targetSize,
                        arguments: new[] {AntiRingingStrength, offset.X, offset.Y},
                        linearSampling: true
                    );

                var yuvFilters = new IFilter[] {new YSourceFilter(), new USourceFilter(), new VSourceFilter()};
                return GetJincFilter(shader, yuvFilters).ConvertToRgb();
            }
        }

        public class Jinc2DChromaScaler : RenderChainUi<Jinc2DChroma, Jinc2DConfigDialog>
        {
            protected override string ConfigFileName
            {
                get { return "Mpdn.Jinc2DChroma"; }
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
                        Guid = new Guid("D93E8C6F-1A4C-40D2-913A-3773C00D1541"),
                        Name = "Jinc2D Chroma",
                        Description = "Jinc (cylindrical) chroma upscaler"
                    };
                }
            }
        }
    }
}
