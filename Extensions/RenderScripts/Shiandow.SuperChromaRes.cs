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
using Mpdn.RenderScript.Scaler;
using SharpDX;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Shiandow.SuperRes
    {
        public class SuperChromaRes : RenderChain, IChromaScaler
        {
            #region Settings

            public int Passes { get; set; }
            public float Strength { get; set; }
            public float Softness { get; set; }
            public bool Prescaler { get; set; }

            #endregion

            private readonly IScaler upscaler, downscaler;

            public SuperChromaRes()
            {
                Passes = 1;

                Strength = 1.0f;
                Softness = 0.0f;
                Prescaler = true;

                upscaler = new Bilinear();
                downscaler = new Bilinear();
            }

            protected override string ShaderPath
            {
                get { return @"SuperRes\SuperChromaRes"; }
            }

            private bool IsIntegral(double x)
            {
                return x == Math.Truncate(x);
            }

            public override IFilter CreateFilter(IFilter input)
            {
                var chromaFilter = input as ChromaFilter;
                if (chromaFilter != null)
                    return chromaFilter.MakeNew(this);

                return input;
            }

            public IFilter CreateChromaFilter(IFilter lumaInput, IFilter chromaInput, TextureSize targetSize, Vector2 chromaOffset)
            {
                IFilter hiRes;

                var chromaSize = chromaInput.OutputSize;
                var lumaSize = lumaInput.OutputSize;

                float[] yuvConsts = Renderer.Colorimetric.GetYuvConsts();
                int bitdepth = Renderer.InputFormat.GetBitDepth();        
                bool limited = Renderer.Colorimetric.IsLimitedRange();                

                Vector2 offset = Renderer.ChromaOffset;
                Vector2 adjointOffset = -offset * lumaSize / chromaSize;

                string superResMacros = "";
                if (IsIntegral(Strength))
                    superResMacros += String.Format("strength = {0};", Strength);
                if (IsIntegral(Softness))
                    superResMacros += String.Format("softness = {0};", Softness);

                string diffMacros = string.Format("LimitedRange = {0}; range = {1}", limited ? 1 : 0, (1 << bitdepth) - 1);

                var CopyLuma = CompileShader("CopyLuma.hlsl");
                var CopyChroma = CompileShader("CopyChroma.hlsl");
                var MergeChroma = CompileShader("MergeChroma.hlsl").Configure(format: TextureFormat.Float16);

                var Diff = CompileShader("Diff.hlsl", macroDefinitions: diffMacros)
                    .Configure(arguments: yuvConsts, format: TextureFormat.Float16);

                var SuperRes = CompileShader("SuperResEx.hlsl", macroDefinitions: superResMacros)
                    .Configure( 
                        arguments: new[] { Strength, Softness, yuvConsts[0], yuvConsts[1], offset.X, offset.Y }
                    );

                var CrossBilateral = CompileShader("CrossBilateral.hlsl")
                    .Configure( 
                        arguments: new[] { offset.X, offset.Y, yuvConsts[0], yuvConsts[1] },
                        perTextureLinearSampling: new[] { true, false }
                    );

                var GammaToLab = CompileShader("../../Common/GammaToLab.hlsl");
                var LabToGamma = CompileShader("../../Common/LabToGamma.hlsl");
                var LinearToGamma = CompileShader("../../Common/LinearToGamma.hlsl");
                var GammaToLinear = CompileShader("../../Common/GammaToLinear.hlsl");
                var LabToLinear = CompileShader("../../Common/LabToLinear.hlsl");
                var LinearToLab = CompileShader("../../Common/LinearToLab.hlsl");

                hiRes = Prescaler 
                    ? (IFilter) new ShaderFilter(CrossBilateral, lumaInput, chromaInput) 
                    : (IFilter) new ChromaFilter(lumaInput, chromaInput);

                for (int i = 1; i <= Passes; i++)
                {
                    IFilter diff, linear;

                    // Compare to chroma
                    linear = new ShaderFilter(GammaToLinear, hiRes.ConvertToRgb());
                    linear = new ResizeFilter(linear, chromaSize, adjointOffset, upscaler, downscaler);
                    diff = new ShaderFilter(Diff, linear, chromaInput);

                    // Update result
                    hiRes = new ShaderFilter(SuperRes, hiRes, diff);
                }

                return hiRes;
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
