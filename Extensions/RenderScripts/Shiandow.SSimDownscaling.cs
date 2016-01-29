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
using Mpdn.Extensions.Framework.RenderChain;
using Mpdn.RenderScript;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Shiandow.SSimDownscaling
    {
        public class SSimDownscaler : RenderChain
        {
		    public static class ShaderFilterHelper
		    {
		        public static ITextureFilter ApplyTo<T>(this IShaderFilterSettings<T> settings, params IBaseTextureFilter[] inputFilters)
		            where T : IShaderBase
		        {
		            if (settings is IShaderFilterSettings<IShader>)
		                return new ShaderFilter((IShaderFilterSettings<IShader>)settings, inputFilters);
		            if (settings is IShaderFilterSettings<IShader11>)
		                return new Shader11Filter((IShaderFilterSettings<IShader11>)settings, inputFilters);

		            throw new ArgumentException("Unsupported Shader type.");
		        }

		        public static ITextureFilter ApplyTo<T>(this IShaderFilterSettings<T> settings, IEnumerable<IBaseTextureFilter> inputFilters)
		            where T : IShaderBase
		        {
		            return settings.ApplyTo(inputFilters.ToArray());
		        }

		        public static ITextureFilter Apply<T>(this ITextureFilter filter, IShaderFilterSettings<T> settings)
		            where T : IShaderBase
		        {
		            return settings.ApplyTo(filter);
		        }

		        public static ITextureFilter ApplyTo<T>(this T shader, params IBaseTextureFilter[] inputFilters)
		            where T : IShaderBase
		        {
		            if (shader is IShader)
		                return new ShaderFilter((IShader)shader, inputFilters);
		            if (shader is IShader11)
		                return new Shader11Filter((IShader11)shader, inputFilters);

		            throw new ArgumentException("Unsupported Shader type.");
		        }

		        public static ITextureFilter ApplyTo<T>(this T shader, IEnumerable<IBaseTextureFilter> inputFilters)
		            where T : IShaderBase
		        {
		            return shader.ApplyTo(inputFilters.ToArray());
		        }

		        public static ITextureFilter Apply<T>(this ITextureFilter filter, T shader)
		            where T : IShaderBase
		        {
		            return shader.ApplyTo(filter);
		        }
		    }

            private Tuple<ITextureFilter, ITextureFilter> DownscaledVariance(ITextureFilter input, TextureSize targetSize)
            {
                var HDownscaler = CompileShader("Scalers/Downscaler.hlsl", macroDefinitions: "axis = 0;")
                    .Configure(transform: s => new TextureSize(targetSize.Width, s.Height), format: input.Output.Format);
                var VDownscaler = CompileShader("Scalers/Downscaler.hlsl", macroDefinitions: "axis = 1;")
                    .Configure(transform: s => new TextureSize(s.Width, targetSize.Height), format: input.Output.Format);
                var HVar = CompileShader("Scalers/DownscaledVar_H.hlsl")
                    .Configure(transform: s => new TextureSize(targetSize.Width, s.Height), format: input.Output.Format);
                var VVar = CompileShader("Scalers/DownscaledVar_V.hlsl")
                    .Configure(transform: s => new TextureSize(s.Width, targetSize.Height), format: input.Output.Format);

                var hMean = HDownscaler.ApplyTo(input);
                var mean = VDownscaler.ApplyTo(hMean);

                var hVariance = HVar.ApplyTo(input, hMean);
                var variance = VVar.ApplyTo(hVariance, hMean, mean);

                return new Tuple<ITextureFilter, ITextureFilter>(mean, variance);
            }

            private Tuple<ITextureFilter, ITextureFilter> ConvolvedVariance(ITextureFilter input)
            {
                var HConvolver = CompileShader("Scalers/Convolver.hlsl", macroDefinitions: "axis = 0;").Configure(format: input.Output.Format);
                var VConvolver = CompileShader("Scalers/Convolver.hlsl", macroDefinitions: "axis = 1;").Configure(format: input.Output.Format);
                var HVar = CompileShader("Scalers/ConvolvedVar_H.hlsl").Configure(format: input.Output.Format);
                var VVar = CompileShader("Scalers/ConvolvedVar_V.hlsl").Configure(format: input.Output.Format);

                var hMean = HConvolver.ApplyTo(input);
                var mean = VConvolver.ApplyTo(hMean);

                var hVariance = HVar.ApplyTo(input, hMean);
                var variance = VVar.ApplyTo(hVariance, hMean, mean);

                return new Tuple<ITextureFilter, ITextureFilter>(mean, variance);
            }

            private ITextureFilter Downscale(ITextureFilter input, TextureSize targetSize)
            {
                var HDownscaler = CompileShader("Scalers/Downscaler.hlsl", macroDefinitions: "axis = 0;")
                    .Configure(transform: s => new TextureSize(targetSize.Width, s.Height), format: input.Output.Format);
                var VDownscaler = CompileShader("Scalers/Downscaler.hlsl", macroDefinitions: "axis = 1;")
                    .Configure(transform: s => new TextureSize(s.Width, targetSize.Height), format: input.Output.Format);

                var hMean = HDownscaler.ApplyTo(input);
                var mean = VDownscaler.ApplyTo(hMean);

                return mean;
            }

            private ITextureFilter Convolve(ITextureFilter input)
            {
                var HConvolver = CompileShader("Scalers/Convolver.hlsl", macroDefinitions: "axis = 0;").Configure(format: input.Output.Format);
                var VConvolver = CompileShader("Scalers/Convolver.hlsl", macroDefinitions: "axis = 1;").Configure(format: input.Output.Format);

                var hMean = HConvolver.ApplyTo(input);
                var mean = VConvolver.ApplyTo(hMean);

                return mean;
            }

            protected override ITextureFilter CreateFilter(ITextureFilter input)
            {
                var targetSize = Renderer.TargetSize;

                if (!IsDownscalingFrom(input))
                    return input;

                var CalcR = CompileShader("calcR.hlsl").Configure(format: TextureFormat.Float16);
                var calcT = CompileShader("calcT.hlsl").Configure(format: TextureFormat.Float16);
                var CalcD = CompileShader("calcD.hlsl");

                var H = input;

                var L = DownscaledVariance(H, targetSize);
                var M = ConvolvedVariance(L.Item1);

                var Sl = M.Item2;
                var Sh = Convolve(L.Item2);

                var R = CalcR.ApplyTo(Sh, Sl, M.Item1); // R = sqrt(1 + Sh / Sl)
                var T = calcT.ApplyTo(R, M.Item1); // T = (1-R)*M

                var Tc = Convolve(T);
                var Rc = Convolve(R);

                return CalcD.ApplyTo(Tc, Rc, L.Item1); // D = Tc + Rc*L;;
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
