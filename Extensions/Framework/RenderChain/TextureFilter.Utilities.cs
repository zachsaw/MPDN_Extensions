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
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Mpdn.Extensions.Framework.Filter;
using Mpdn.RenderScript;
using SharpDX;

namespace Mpdn.Extensions.Framework.RenderChain
{
    public abstract class BasicFilter : TextureFilter
    {
        protected BasicFilter(ITextureFilter inputFilter)
            : base(inputFilter)
        { }

        protected abstract void Render(ITexture2D input);

        protected override void Render(IList<ITextureOutput<IBaseTexture>> inputs)
        {
            var texture = inputs
                .Select(x => x.Texture)
                .OfType<ITexture2D>()
                .SingleOrDefault();

            if (texture == null)
                return;

            Render(texture);
        }

        protected override TextureSize OutputSize
        {
            get { return InputFilters[0].Output.Size; }
        }

        protected override TextureFormat OutputFormat
        {
            get { return InputFilters[0].Output.Format; }
        }
    }

    public sealed class RgbFilter : BasicFilter
    {
        public readonly YuvColorimetric Colorimetric;
        public readonly bool OutputLimitedRange;
        public readonly bool OutputLimitChroma;

        public RgbFilter(ITextureFilter inputFilter, bool limitedRange)
            : this(inputFilter, null, limitedRange)
        {
        }

        public RgbFilter(ITextureFilter inputFilter, YuvColorimetric? colorimetric = null, bool? limitedRange = null, bool? limitChroma = null)
            : base(inputFilter)
        {
            Colorimetric = colorimetric ?? Renderer.Colorimetric;
            OutputLimitedRange = limitedRange ?? Renderer.OutputLimitedRange;
            OutputLimitChroma = limitChroma ?? Renderer.LimitChroma;
        }

        protected override IFilter<ITextureOutput<ITexture2D>> Optimize()
        {
            var input = InputFilters[0] as YuvFilter;
            if (input != null && input.Colorimetric == Colorimetric && input.OutputLimitedRange == OutputLimitedRange)
                return (ITextureFilter) input.InputFilters[0];

            return this;
        }

        protected override void Render(ITexture2D input)
        {
            Renderer.ConvertToRgb(Target.Texture, input, Colorimetric, OutputLimitedRange, OutputLimitChroma);
        }
    }

    public sealed class YuvFilter : BasicFilter
    {
        public readonly YuvColorimetric Colorimetric;
        public readonly bool OutputLimitedRange;

        public YuvFilter(ITextureFilter inputFilter, bool limitedRange)
            : this(inputFilter, null, limitedRange)
        {
        }

        public YuvFilter(ITextureFilter inputFilter, YuvColorimetric? colorimetric = null, bool? limitedRange = null)
            : base(inputFilter)
        {
            Colorimetric = colorimetric ?? Renderer.Colorimetric;
            OutputLimitedRange = limitedRange ?? Renderer.OutputLimitedRange;
        }

        protected override IFilter<ITextureOutput<ITexture2D>> Optimize()
        {
            var input = InputFilters[0] as RgbFilter;
            if (input != null && input.Colorimetric == Colorimetric && input.OutputLimitedRange == OutputLimitedRange)
                return (ITextureFilter) input.InputFilters[0];

            var sourceFilter = InputFilters[0] as VideoSourceFilter.TrueSourceFilter;
            if (sourceFilter != null)
                return sourceFilter.GetYuv();

            return this;
        }

        protected override void Render(ITexture2D input)
        {
            Renderer.ConvertToYuv(Target.Texture, input, Colorimetric, OutputLimitedRange);
        }
    }

    public sealed class ChromaSourceFilter : ShaderFilter
    {
        public ChromaSourceFilter()
            : base(GetShader(), new USourceFilter(), new VSourceFilter())
        {
        }

        private static IShader GetShader()
        {
            var asmPath = typeof (IRenderScript).Assembly.Location;
            var shaderDataFilePath =
                Path.Combine(PathHelper.GetDirectoryName(asmPath),
                    "Extensions", "RenderScripts", "Common");
            return ShaderCache.CompileShader(Path.Combine(shaderDataFilePath, "MergeChromaYZFromSource.hlsl"));
        }
    }

    public class ResizeFilter : TextureFilter, IResizeableFilter
    {
        private TextureSize m_OutputSize;
        private readonly TextureChannels m_Channels;
        private readonly Vector2 m_Offset;

        private readonly IScaler m_Downscaler;
        private readonly IScaler m_Upscaler;
        private IScaler m_Convolver;
        private bool m_Tagged;

        public ResizeFilter(ITextureFilter<ITexture2D> inputFilter, 
            IScaler upscaler = null, IScaler downscaler = null, IScaler convolver = null)
            : this(inputFilter, inputFilter.Output.Size, TextureChannels.All, Vector2.Zero, upscaler, downscaler, convolver)
        { }

        public ResizeFilter(ITextureFilter<ITexture2D> inputFilter, TextureSize outputSize,
            IScaler upscaler = null, IScaler downscaler = null, IScaler convolver = null)
            : this(inputFilter, outputSize, TextureChannels.All, Vector2.Zero, upscaler, downscaler, convolver)
        { }

        public ResizeFilter(ITextureFilter<ITexture2D> inputFilter, TextureSize outputSize, TextureChannels channels, 
            IScaler upscaler = null, IScaler downscaler = null, IScaler convolver = null)
            : this(inputFilter, outputSize, channels, Vector2.Zero, upscaler, downscaler, convolver)
        { }

        public ResizeFilter(ITextureFilter<ITexture2D> inputFilter, TextureSize outputSize, Vector2 offset, 
            IScaler upscaler = null, IScaler downscaler = null, IScaler convolver = null)
            : this(inputFilter, outputSize, TextureChannels.All, offset, upscaler, downscaler, convolver)
        { }

        public ResizeFilter(ITextureFilter<ITexture2D> inputFilter, TextureSize outputSize, TextureChannels channels, Vector2 offset, 
            IScaler upscaler = null, IScaler downscaler = null, IScaler convolver = null)
            : base(inputFilter)
        {
            m_OutputSize = outputSize;
            m_Channels = channels;
            m_Offset = offset;

            m_Upscaler = upscaler ?? Renderer.LumaUpscaler;
            m_Downscaler = downscaler ?? Renderer.LumaDownscaler;
            m_Convolver = convolver;
        }

        private ITextureFilter<ITexture2D> InputFilter
        {
            get { return (ITextureFilter<ITexture2D>)InputFilters[0]; }
        }

        public void EnableTag() 
        {
            m_Tagged = true;
        }

        public void ForceOffsetCorrection()
        {
            if (!m_Offset.IsZero)
                m_Convolver = m_Convolver ?? m_Upscaler;
        }

        public void SetSize(TextureSize targetSize)
        {
            m_OutputSize = targetSize;
        }

        protected override IFilter<ITextureOutput<ITexture2D>> Optimize()
        {
            if (InputFilter.Output.Size == m_OutputSize && m_Convolver == null)
                return InputFilter;

            if (m_Tagged)
                AddTag(Status());

            return this;
        }

        public string Status()
        {
            var inputSize = InputFilters[0].Output.Size;
            return StatusHelpers.ScaleDescription(inputSize, OutputSize, m_Upscaler, m_Downscaler, m_Convolver);            
        }

        protected override TextureSize OutputSize
        {
            get { return m_OutputSize; }
        }

        protected override TextureFormat OutputFormat
        {
            get { return InputFilter.Output.Format; }
        }

        protected override void Render(IList<ITextureOutput<IBaseTexture>> inputs)
        {
            var texture = inputs
                .Select(x => x.Texture)
                .OfType<ITexture2D>()
                .SingleOrDefault();
            if (texture == null)
                return;

            Renderer.Scale(Target.Texture, texture, m_Channels, m_Offset, m_Upscaler, m_Downscaler, m_Convolver);
        }
    }

    public sealed class MergeFilter : ShaderFilter
    {
        public MergeFilter(ITextureFilter inputY, ITextureFilter inputUv)
            : base(GetShader(true), inputY, inputUv)
        {
        }

        public MergeFilter(ITextureFilter inputY, ITextureFilter inputU, ITextureFilter inputV)
            : base(GetShader(false), inputY, inputU, inputV)
        {
        }

        private static IShader GetShader(bool mergedUv)
        {
            var asmPath = typeof(IRenderScript).Assembly.Location;
            var shaderDataFilePath =
                Path.Combine(PathHelper.GetDirectoryName(asmPath),
                    "Extensions", "RenderScripts", "Common");
            var shaderFile = mergedUv ? "MergeY_UV.hlsl" : "MergeY_U_V.hlsl";
            return ShaderCache.CompileShader(Path.Combine(shaderDataFilePath, shaderFile));
        }
    }

    public static class ConversionHelper
    {
        public static ITextureFilter ConvertToRgb(this ITextureFilter filter)
        {
            return new RgbFilter(filter);
        }

        public static ITextureFilter ConvertToYuv(this ITextureFilter filter)
        {
            return new YuvFilter(filter);
        }

        public static IResizeableFilter Transform(this IResizeableFilter filter, Func<ITextureFilter, ITextureFilter> transformation)
        {
            return new TransformedResizeableFilter(transformation, filter);
        }

        public static ITextureFilter Apply(this ITextureFilter filter, Func<ITextureFilter, ITextureFilter> map)
        {
            return map(filter);
        }

        public static ITextureFilter AddTaggedResizer(this ITextureFilter<ITexture2D> filter)
        {
            return filter.SetSize(filter.Output.Size, true);
        }

        public static ITextureFilter SetSize(this ITextureFilter<ITexture2D> filter, TextureSize size, bool tagged = false)
        {
            var resizeable = (filter as IResizeableFilter) ?? new ResizeFilter(filter);
            if (tagged)
                resizeable.EnableTag();
            resizeable.SetSize(size);
            return resizeable;
        }

        #region Auxilary class(es)

        private sealed class TransformedResizeableFilter : TextureFilter, IResizeableFilter
        {
            private readonly IResizeableFilter m_InputFilter;
            private readonly Func<ITextureFilter, ITextureFilter> m_Transformation;

            public TransformedResizeableFilter(Func<ITextureFilter, ITextureFilter> transformation, IResizeableFilter inputFilter)
            {
                m_InputFilter = inputFilter;
                m_Transformation = transformation;
            }

            protected override IFilter<ITextureOutput<ITexture2D>> Optimize()
            {
                var result = m_Transformation(m_InputFilter);

                if (m_InputFilter.Output.Size != result.Output.Size)
                    throw new InvalidOperationException("Transformation is not allowed to change the size.");
               
                return m_Transformation(m_InputFilter);
            }

            public void EnableTag() 
            {
                m_InputFilter.EnableTag();
            }

            public void SetSize(TextureSize outputSize)
            {
                m_InputFilter.SetSize(outputSize);
            }

            protected override TextureSize OutputSize
            {
                get { return m_InputFilter.Output.Size; }
            }

            protected override TextureFormat OutputFormat
            {
                get { return m_InputFilter.Output.Format; }
            }

            protected override void Render(IList<ITextureOutput<IBaseTexture>> textureOutputs)
            {
                throw new NotImplementedException();
            }
        }

        #endregion
    }
}
