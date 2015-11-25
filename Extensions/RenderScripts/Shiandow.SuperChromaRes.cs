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

            private readonly IScaler m_Upscaler, m_Downscaler;

            public SuperChromaRes()
            {
                Passes = 1;

                Strength = 1.0f;
                Softness = 0.0f;
                Prescaler = true;

                m_Upscaler = new Bilinear();
                m_Downscaler = new Bilinear();
            }

            protected override string ShaderPath
            {
                get { return @"SuperRes\SuperChromaRes"; }
            }

            private bool IsIntegral(double x)
            {
                return Math.Abs(x - Math.Truncate(x)) < 0.005;
            }

            protected override ITextureFilter CreateFilter(ITextureFilter input)
            {
                var chromaFilter = input as ChromaFilter;
                if (chromaFilter == null)
                    return input;

                var bilateral = Prescaler ? new Bilateral.Bilateral() : IdentityChain;
                input += bilateral;

                var chromaScaler = new SuperChromaResScaler((ChromaFilter)input, this);
                
                return input + chromaScaler;
            }

            public virtual ITextureFilter CreateChromaFilter(ITextureFilter lumaInput, ITextureFilter chromaInput, TextureSize targetSize, Vector2 chromaOffset)
            {
                var input = new ChromaFilter(lumaInput, chromaInput, targetSize, chromaOffset);
                return Process(input);
            }

            private class SuperChromaResScaler: SuperChromaRes
            {
                private readonly ChromaFilter m_InitialInput;

                public override string Status { get { return "SuperChromaRes"; } }

                public SuperChromaResScaler(ChromaFilter initialInput, SuperChromaRes parent)
                {
                    m_InitialInput = initialInput;

                    Passes = parent.Passes;
                    Strength = parent.Strength;
                    Softness = parent.Softness;
                }

                protected override ITextureFilter CreateFilter(ITextureFilter input)
                {
                    return this.MakeChromaFilter(input);
                }

                public override ITextureFilter CreateChromaFilter(ITextureFilter lumaInput, ITextureFilter chromaInput, TextureSize targetSize, Vector2 chromaOffset)
                {
                    var chromaSize = chromaInput.Output.Size;
                    var lumaSize = lumaInput.Output.Size;

                    float[] yuvConsts = Renderer.Colorimetric.GetYuvConsts();
                    int bitdepth = Renderer.InputFormat.GetBitDepth();
                    bool limited = Renderer.Colorimetric.IsLimitedRange();

                    Vector2 adjointOffset = -chromaOffset * lumaSize / chromaSize;

                    string superResMacros = "";
                    if (IsIntegral(Strength))
                        superResMacros += string.Format("strength = {0};", Strength);
                    if (IsIntegral(Softness))
                        superResMacros += string.Format("softness = {0};", Softness);

                    string diffMacros = string.Format("LimitedRange = {0}; range = {1}", limited ? 1 : 0, (1 << bitdepth) - 1);

                    var Diff = CompileShader("Diff.hlsl", macroDefinitions: diffMacros)
                        .Configure(arguments: yuvConsts, format: TextureFormat.Float16);

                    var SuperRes = CompileShader("SuperResEx.hlsl", macroDefinitions: superResMacros)
                        .Configure(
                            arguments: new[] { Strength, Softness, yuvConsts[0], yuvConsts[1], chromaOffset.X, chromaOffset.Y }
                        );

                    var LinearToGamma = CompileShader("../../Common/LinearToGamma.hlsl");
                    var GammaToLinear = CompileShader("../../Common/GammaToLinear.hlsl");

                    if (Passes == 0) return m_InitialInput;

                    m_InitialInput.SetSize(lumaSize);
                    var hiRes = m_InitialInput.ConvertToYuv();

                    for (int i = 1; i <= Passes; i++)
                    {
                        ITextureFilter diff, linear;

                        // Compare to chroma
                        linear = new ShaderFilter(GammaToLinear, hiRes.ConvertToRgb());
                        linear = new ResizeFilter(linear, chromaSize, adjointOffset, m_Upscaler, m_Downscaler);
                        diff = new ShaderFilter(Diff, linear, chromaInput);

                        // Update result
                        hiRes = new ShaderFilter(SuperRes, hiRes, diff);
                    }

                    return hiRes.ConvertToRgb();
                }
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
