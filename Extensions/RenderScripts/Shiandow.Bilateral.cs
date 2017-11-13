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
        public enum BilateralMode
        {
            Legacy=0,
            Krig=1
        }

        public class Bilateral : ChromaChain
        {
            #region Settings

            public float Strength { get; set; }
            public BilateralMode Mode { get; set; }

            public Bilateral()
            {
                Strength = 0.5f;
                Mode = BilateralMode.Krig;
            }

            #endregion

            private ITextureFilter DownscaleLuma(ITextureFilter luma, ITextureFilter chroma, TextureSize targetSize, Vector2 adjointOffset)
            {
                var HDownscaler = new Shader(FromFile("LumaDownscalerI.hlsl", compilerOptions: "axis = 0;"))
                {
                    Transform = s => new TextureSize(targetSize.Width, s.Height),
                    Arguments = new ArgumentList { adjointOffset },
                    Format = TextureFormat.Float16_RG
                };
                var VDownscaler = new Shader(FromFile("LumaDownscalerII.hlsl", compilerOptions: "axis = 1;"))
                {
                    Transform = s => new TextureSize(s.Width, targetSize.Height),
                    Arguments = new ArgumentList { adjointOffset },
                    Format = TextureFormat.Float16
                };

                var Y = HDownscaler.ApplyTo(luma);
                var YUV = VDownscaler.ApplyTo(Y, chroma);

                return YUV;
            }

            public override ITextureFilter ScaleChroma(ICompositionFilter composition)
            {
                var lumaSize = composition.Luma.Size();
                var chromaSize = composition.Chroma.Size();
                var targetSize = composition.TargetSize;

                float[] yuvConsts = Renderer.Colorimetric.GetYuvConsts();
                var chromaOffset = composition.ChromaOffset;

                // Fall back to default when downscaling is needed
                if ((chromaSize > targetSize).Any || chromaSize == targetSize)
                    return composition;

                Vector2 adjointOffset = -(chromaOffset * lumaSize) / chromaSize;

                var crossBilateral = new Shader(FromFile((Mode == BilateralMode.Legacy) ? "CrossBilateral.hlsl" : "KrigBilateral.hlsl"));
                crossBilateral["chromaParams"] = new Vector4(chromaOffset.X, chromaOffset.Y, yuvConsts[0], yuvConsts[1] );
                crossBilateral["power"] = Strength;

                var resizedLuma = composition.Luma.SetSize(composition.TargetSize, tagged: true);
                var lowresYuv = DownscaleLuma(composition.Luma, composition.Chroma, chromaSize, adjointOffset);

                return crossBilateral.ApplyTo(resizedLuma, lowresYuv).ConvertToRgb();
            }
        }
        
        public class BilateralUi : RenderChainUi<Bilateral, BilateralConfigDialog>
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
