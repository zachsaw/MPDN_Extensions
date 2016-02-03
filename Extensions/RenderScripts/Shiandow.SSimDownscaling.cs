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

// See also: Perceptually Based Downscaling of Images, by Oztireli, A. Cengiz and Gross, Markus, 10.1145/2766891, https://graphics.ethz.ch/~cengizo/Files/Sig15PerceptualDownscaling.pdf

using System;
using System.Collections.Generic;
using System.Linq;
using Mpdn.Extensions.Framework.RenderChain;
using Mpdn.RenderScript;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Shiandow.SSimDownscaling
    {
        public static class ShaderFilterHelper
        {
            public static IFilter ApplyTo<T>(this ShaderFilterSettings<T> settings, params IFilter<IBaseTexture>[] inputFilters)
                where T : IShaderBase
            {
                if (settings is ShaderFilterSettings<IShader>)
                    return new ShaderFilter(settings as ShaderFilterSettings<IShader>, inputFilters);
                if (settings is ShaderFilterSettings<IShader11>)
                    return new Shader11Filter(settings as ShaderFilterSettings<IShader11>, inputFilters);

                throw new ArgumentException("Unsupported Shader type.");
            }

            public static IFilter ApplyTo<T>(this ShaderFilterSettings<T> settings, IEnumerable<IFilter<IBaseTexture>> inputFilters)
                where T : IShaderBase
            {
                return settings.ApplyTo(inputFilters.ToArray());
            }

            public static IFilter Apply<T>(this IFilter filter, ShaderFilterSettings<T> settings)
                where T : IShaderBase
            {
                return settings.ApplyTo(filter);
            }

            public static IFilter ApplyTo<T>(this T shader, params IFilter<IBaseTexture>[] inputFilters)
                where T : IShaderBase
            {
                if (shader is IShader)
                    return new ShaderFilter((IShader)shader, inputFilters);
                if (shader is IShader11)
                    return new Shader11Filter((IShader11)shader, inputFilters);

                throw new ArgumentException("Unsupported Shader type.");
            }

            public static IFilter ApplyTo<T>(this T shader, IEnumerable<IFilter<IBaseTexture>> inputFilters)
                where T : IShaderBase
            {
                return shader.ApplyTo(inputFilters.ToArray());
            }

            public static IFilter Apply<T>(this IFilter filter, T shader)
                where T : IShaderBase
            {
                return shader.ApplyTo(filter);
            }
        }

        public class SSimDownscaler : RenderChain
        {
            private void DownscaleAndCalcVar(IFilter input, TextureSize targetSize, out IFilter mean, out IFilter var)
            {
                var HDownscaler = CompileShader("Scalers/Downscaler.hlsl", macroDefinitions: "axis = 0;")
                    .Configure(transform: s => new TextureSize(targetSize.Width, s.Height), format: input.OutputFormat);
                var VDownscaler = CompileShader("Scalers/Downscaler.hlsl", macroDefinitions: "axis = 1;")
                    .Configure(transform: s => new TextureSize(s.Width, targetSize.Height), format: input.OutputFormat);
                var HVar = CompileShader("DownscaledVarI.hlsl", macroDefinitions: "axis = 0;")
                    .Configure(transform: s => new TextureSize(targetSize.Width, s.Height), format: input.OutputFormat);
                var VVar = CompileShader("DownscaledVarII.hlsl", macroDefinitions: "axis = 1;")
                    .Configure(transform: s => new TextureSize(s.Width, targetSize.Height), format: input.OutputFormat);

                var hMean = HDownscaler.ApplyTo(input);
                mean = VDownscaler.ApplyTo(hMean);

                var hVariance = HVar.ApplyTo(input, hMean);
                var = VVar.ApplyTo(hVariance, hMean, mean);
            }

            private void ConvolveAndCalcR(IFilter input, IFilter sh, out IFilter mean, out IFilter r)
            {
                var Convolver = CompileShader("SinglePassConvolver.hlsl").Configure(format: input.OutputFormat);
                var CalcR = CompileShader("CalcR.hlsl").Configure(format: TextureFormat.Float16);

                mean = input.Apply(Convolver);
                r = CalcR.ApplyTo(input, mean, sh);
            }

            protected override IFilter CreateFilter(IFilter input)
            {
                IFilter H = input, Sh, L, M, R;
                var targetSize = Renderer.TargetSize;

                if (!IsDownscalingFrom(input))
                    return input;

                var Calc = CompileShader("calc.hlsl");

                DownscaleAndCalcVar(H, targetSize, out L, out Sh);
                ConvolveAndCalcR(L, Sh, out M, out R);

                return Calc.ApplyTo(R, M, L);
            }
        }

        public class SSimDownscalerUi : RenderChainUi<SSimDownscaler>
        {
            protected override string ConfigFileName
            {
                get { return "Structural Similarity Based Downscaling"; }
            }

            public override string Category
            {
                get { return "Downscaling"; }
            }

            public override ExtensionUiDescriptor Descriptor
            {
                get
                {
                    return new ExtensionUiDescriptor
                    {
                        Guid = new Guid("ED1BD188-BA46-11E5-BADB-8BEA19563991"),
                        Name = "SSIM downscaler",
                        Description = "Structural Similarity based Downscaling",
                        Copyright = "Shiandow",
                    };
                }
            }
        }
    }
}