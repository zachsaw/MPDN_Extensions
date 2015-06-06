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
using Mpdn.Extensions.Framework.Filter;
using Mpdn.Extensions.Framework.RenderChain;
using Mpdn.RenderScript;
using Mpdn.RenderScript.Scaler;
using SharpDX;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Shiandow.SuperRes
    {
        public class SuperChromaRes : RenderChain
        {
            #region Settings

            public int Passes { get; set; }

            public float Strength { get; set; }
            public float Sharpness { get; set; }
            public float AntiAliasing { get; set; }
            public float AntiRinging { get; set; }
            public float Softness { get; set; }

            public bool FirstPassOnly;

            #endregion

            private readonly IScaler upscaler, downscaler;

            public SuperChromaRes()
            {
                Passes = 2;

                Strength = 0.75f;
                Sharpness = 0.25f;
                AntiAliasing = 0.25f;
                AntiRinging = 0.0f;
                Softness = 0.5f;

                FirstPassOnly = false; /* Not used anyway */
                upscaler = new Bilinear();
                downscaler = new Bilinear();
            }

            protected override string ShaderPath
            {
                get { return @"SuperRes\SuperChromaRes"; }
            }

            public override IFilter CreateFilter(IFilter input)
            {
                IFilter gamma, chroma;

                var chromaSize = (TextureSize)Renderer.ChromaSize;
                var targetSize = input.OutputSize;

                // Original values
                var uInput = new USourceFilter();
                var vInput = new VSourceFilter();

                float[] yuvConsts = new float[0];
                switch (Renderer.Colorimetric)
                {
                    case YuvColorimetric.Auto : return input;
                    case YuvColorimetric.FullRangePc601: yuvConsts = new[] { 0.114f, 0.299f, 0.0f}; break;
                    case YuvColorimetric.FullRangePc709: yuvConsts = new[] { 0.0722f, 0.2126f, 0.0f }; break;
                    case YuvColorimetric.FullRangePc2020: yuvConsts = new[] { 0.0593f, 0.2627f, 0.0f }; break;
                    case YuvColorimetric.ItuBt601: yuvConsts = new[] { 0.114f, 0.299f, 1.0f }; break;
                    case YuvColorimetric.ItuBt709: yuvConsts = new[] { 0.0722f, 0.2126f, 1.0f }; break;
                    case YuvColorimetric.ItuBt2020: yuvConsts = new[] { 0.0593f, 0.2627f, 1.0f }; break;
                }

                // Skip if downscaling
                if (targetSize.Width <= chromaSize.Width && targetSize.Height <= chromaSize.Height)
                    return input;

                Vector2 offset = Renderer.ChromaOffset;
                Vector2 adjointOffset = -offset * targetSize / chromaSize;

                var consts = new[] { Strength, Sharpness   , AntiAliasing, AntiRinging,  
                                     Softness, yuvConsts[0], yuvConsts[1], 0.0f,
                                     offset.X, offset.Y };

                var CopyLuma = CompileShader("CopyLuma.hlsl");
                var CopyChroma = CompileShader("CopyChroma.hlsl");
                var MergeChroma = CompileShader("MergeChroma.hlsl").Configure( format: TextureFormat.Float16 );
                var Diff = CompileShader("Diff.hlsl").Configure(arguments: yuvConsts, format: TextureFormat.Float16);
                var SuperRes = CompileShader("SuperRes.hlsl", macroDefinitions:
                        (AntiRinging == 0 ? "SkipAntiRinging = 1;" : "") +
                        (AntiAliasing == 0 ? "SkipAntiAliasing = 1;" : "") +
                        (Sharpness == 0 ? "SkipSharpening = 1;" : "") +
                        (Softness  == 0 ? "SkipSoftening = 1;" : "")
                        ).Configure(arguments: consts);

                var GammaToLab = CompileShader("../../Common/GammaToLab.hlsl");
                var LabToGamma = CompileShader("../../Common/LabToGamma.hlsl");
                var LinearToGamma = CompileShader("../../Common/LinearToGamma.hlsl");
                var GammaToLinear = CompileShader("../../Common/GammaToLinear.hlsl");
                var LabToLinear = CompileShader("../../Common/LabToLinear.hlsl");
                var LinearToLab = CompileShader("../../Common/LinearToLab.hlsl");

                chroma = new ShaderFilter(MergeChroma, uInput, vInput);
                chroma = new RgbFilter(chroma, limitChroma: false);
                gamma = input;

                for (int i = 1; i <= Passes; i++)
                {
                    IFilter loRes, diff, linear;
                    bool useBilinear = (upscaler is Bilinear) || (FirstPassOnly && i != 1);

                    // Compare to chroma
                    linear = new ShaderFilter(GammaToLinear, gamma);
                    loRes = new ResizeFilter(linear, chromaSize, adjointOffset, upscaler, downscaler);
                    diff = new ShaderFilter(Diff, loRes, chroma);
                    if (!useBilinear)
                        diff = new ResizeFilter(diff, targetSize, offset, upscaler, downscaler); // Scale to output size

                    // Update result
                    gamma = new ShaderFilter( 
                        SuperRes.Configure( perTextureLinearSampling: new [] { false, useBilinear, false } ),
                        gamma, diff, chroma);
                }

                return gamma;
            }
        }

        public class SuperChromaResUi : RenderChainUi<SuperChromaRes, SuperChromaResConfigDialog>
        {
            protected override string ConfigFileName
            {
                get { return "SuperChromaRes"; }
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
                        Guid = new Guid("AC6F46E2-C04E-4A20-AF68-EFA8A6CA7FCD"),
                        Name = "SuperChromaRes",
                        Description = "SuperChromaRes chroma scaling",
                        Copyright = "SuperChromaRes by Shiandow",
                    };
                }
            }
        }
    }
}
