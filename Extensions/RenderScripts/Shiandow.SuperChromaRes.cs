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
using System.Linq;
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
            public bool LegacyDownscaling { get; set; }

            #endregion

            public SuperChromaRes()
            {
                Passes = 1;

                Strength = 1.0f;
                Softness = 0.0f;
                Prescaler = true;

                LegacyDownscaling = false;
            }

            protected override string ShaderPath
            {
                get { return @"SuperRes\SuperChromaRes"; }
            }

            private ITextureFilter DownscaleAndDiff(ITextureFilter input, ITextureFilter original, TextureSize targetSize, Vector2 adjointOffset)
            {
                var HDownscaler = CompileShader("ChromaDownscaler.hlsl", macroDefinitions: "axis = 0;").Configure(
                        transform: s => new TextureSize(targetSize.Width, s.Height),
                        arguments: new ArgumentList { adjointOffset });
                var VDownscaleAndDiff = CompileShader("DownscaleAndDiff.hlsl", macroDefinitions: "axis = 1;").Configure(
                        transform: s => new TextureSize(s.Width, targetSize.Height),
                        arguments: new ArgumentList { adjointOffset },
                        format: TextureFormat.Float16);

                var hMean = HDownscaler.ApplyTo(input);
                var diff = VDownscaleAndDiff.ApplyTo(hMean, original);

                return diff;
            }

            protected override ITextureFilter CreateFilter(ITextureFilter input)
            {
                var compositionFilter = input as ICompositionFilter;
                if (compositionFilter == null)
                    return input;

                if (Prescaler)
                    input = compositionFilter + (new Bilateral.Bilateral());

                var chromaSize = compositionFilter.Chroma.Output.Size;
                var lumaSize = compositionFilter.Luma.Output.Size;
                var chromaOffset = compositionFilter.ChromaOffset;

                var downscaler = new Bicubic(0.75f, false);

                float[] yuvConsts = Renderer.Colorimetric.GetYuvConsts();
                int bitdepth = Renderer.InputFormat.GetBitDepth();
                bool limited = Renderer.Colorimetric.IsLimitedRange();

                Vector2 adjointOffset = -chromaOffset * lumaSize / chromaSize;

                string superResMacros = "";
                if (Softness == 0.0f)
                    superResMacros += "SkipSoftening = 1;";
                string diffMacros = string.Format("LimitedRange = {0}; range = {1}", limited ? 1 : 0, (1 << bitdepth) - 1);

                var configArgs = yuvConsts.Concat(new[] { chromaOffset.X, chromaOffset.Y }).ToArray();

                var Diff = CompileShader("Diff.hlsl", macroDefinitions: diffMacros)
                    .Configure(arguments: configArgs, format: TextureFormat.Float16);
                var SuperRes = CompileShader("SuperRes.hlsl", macroDefinitions: superResMacros)
                    .Configure(arguments: (new[] { Strength, Softness }).Concat(configArgs).ToArray());

                if (Passes == 0 || Strength == 0.0f) return input;

                var hiRes = input.SetSize(lumaSize).ConvertToYuv();

                for (int i = 1; i <= Passes; i++)
                {
                    // Downscale and Subtract
                    ITextureFilter diff;
                    if (LegacyDownscaling)
                    {
                        var lores = hiRes.Resize(chromaSize, TextureChannels.ChromaOnly, adjointOffset, null, downscaler);
                        diff = Diff.ApplyTo(lores, compositionFilter.Chroma);
                    }
                    else
                    { diff = DownscaleAndDiff(hiRes, compositionFilter.Chroma, chromaSize, adjointOffset); }

                    // Update result
                    hiRes = SuperRes.ApplyTo(hiRes, diff);
                }

                return hiRes.ConvertToRgb();
            }

            public ITextureFilter CreateChromaFilter(ITextureFilter lumaInput, ITextureFilter chromaInput, TextureSize targetSize, Vector2 chromaOffset)
            {
                return this.MakeChromaFilter(lumaInput, chromaInput, targetSize, chromaOffset);
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
