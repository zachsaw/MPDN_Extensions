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
using System.IO;
using System.Linq;
using Mpdn.Extensions.Framework.Filter;
using Mpdn.Extensions.Framework.RenderChain.Shaders;
using Mpdn.RenderScript;
using SharpDX;

namespace Mpdn.Extensions.Framework.RenderChain.Filters
{
    // Simple process that doesn't change format or size
    public abstract class BasicProcess : TextureProcess<ITexture2D>
    {
        protected override TextureFormat CalcFormat(TextureFormat inputFormat)
        {
            return inputFormat;
        }

        protected override TextureSize CalcSize(TextureSize inputSize)
        {
            return inputSize;
        }
    }

    public interface ICanUndo<TProcess> : ITextureFilter
        where TProcess : IProcess<ITextureFilter, ITextureFilter>
    {
        ITextureFilter Undo();
    }

    public class UndoFilter<TProcess> : TextureFilter, ICanUndo<TProcess>
        where TProcess : IProcess<ITextureFilter, ITextureFilter>
    {
        private readonly ITextureFilter m_InputFilter;

        public ITextureFilter Undo() { return m_InputFilter; }

        public UndoFilter(ITextureFilter inputFilter, TProcess process)
            : base(process.ApplyTo(inputFilter))
        {
            m_InputFilter = inputFilter;
        }
    }

    public abstract class ColorimetricProcess : BasicProcess
    {
        public readonly YuvColorimetric Colorimetric;
        public readonly bool OutputLimitedRange;
        public readonly bool OutputLimitChroma;

        public ColorimetricProcess(YuvColorimetric? colorimetric = null, bool? limitedRange = null, bool? limitChroma = null)
        {
            Colorimetric = colorimetric ?? Renderer.Colorimetric;
            OutputLimitedRange = limitedRange ?? Renderer.OutputLimitedRange;
            OutputLimitChroma = limitChroma ?? Renderer.LimitChroma;
        }
    }

    public class RgbProcess : ColorimetricProcess
    {
        public RgbProcess(YuvColorimetric? colorimetric = null, bool? limitedRange = null, bool? limitChroma = null)
            : base(colorimetric, limitedRange, limitChroma)
        { }

        protected override void Render(ITexture2D input, ITargetTexture output)
        {
            Renderer.ConvertToRgb(output, input, Colorimetric, OutputLimitedRange, OutputLimitChroma);
        }
    }

    public class YuvProcess : ColorimetricProcess
    {
        public YuvProcess(YuvColorimetric? colorimetric = null, bool? limitedRange = null)
            : base(colorimetric, limitedRange)
        { }

        protected override void Render(ITexture2D input, ITargetTexture output)
        {
            Renderer.ConvertToYuv(output, input, Colorimetric, OutputLimitedRange);
        }
    }

    public static class ColorimetricHelper
    {
        public static ITextureFilter ConvertToRgb(ITextureFilter filter)
        {
            var yuv = filter as ICanUndo<YuvProcess>;
            if (yuv != null)
                return yuv.Undo();

            return new UndoFilter<RgbProcess>(filter, new RgbProcess());
        }

        public static ITextureFilter ConvertToYuv(ITextureFilter filter)
        {
            var rgb = filter as ICanUndo<RgbProcess>;
            if (rgb != null)
                return rgb.Undo();

            var sourceFilter = filter as VideoSourceFilter;
            if (sourceFilter != null)
                return sourceFilter.GetYuv();

            return new UndoFilter<YuvProcess>(filter, new YuvProcess());
        }
    }

    public sealed class ChromaSourceFilter : TextureFilter
    {
        public ChromaSourceFilter()
            : base(GetShader().GetHandle().ApplyTo(new USourceFilter(), new VSourceFilter()))
        { }

        private static IShaderConfig GetShader()
        {
            var shaderDataFilePath = Path.Combine(ShaderCache.ShaderPathRoot, "Common");
            return new Shader(DefinitionHelper.FromFile(Path.Combine(shaderDataFilePath, "MergeChromaYZFromSource.hlsl")));
        }
    }

    public class ResizeFilter : UndoFilter<ResizeFilter.IProcess>, IResizeableFilter, IOffsetFilter
    {
        public ResizeFilter(ITextureFilter inputFilter, TextureSize outputSize, IScaler upscaler = null, IScaler downscaler = null, IScaler convolver = null, TextureFormat? outputFormat = null)
            : this(inputFilter, outputSize, TextureChannels.All, Vector2.Zero, upscaler, downscaler, convolver, outputFormat)
        { }

        public ResizeFilter(ITextureFilter inputFilter, TextureSize outputSize, TextureChannels channels, Vector2 offset, IScaler upscaler = null, IScaler downscaler = null, IScaler convolver = null, TextureFormat? outputFormat = null)
            : this(inputFilter, new Process(outputSize, channels, offset, upscaler, downscaler, convolver, outputFormat))
        { }

        private ResizeFilter(ITextureFilter inputFilter, Process process)
            : base(inputFilter, process)
        {
            m_Process = process;
        }

        #region Process Definition

        public interface IProcess : ITextureProcess<ITextureFilter<ITexture2D>> { }

        private class Process : TextureProcess<ITexture2D>, IProcess
        {
            public Process(TextureSize outputSize, TextureChannels channels, Vector2 offset, IScaler upscaler = null, IScaler downscaler = null, IScaler convolver = null, TextureFormat? outputFormat = null)
            {
                m_OutputSize = outputSize;
                m_Channels = channels;
                m_Offset = offset;

                m_Format = outputFormat;

                m_Upscaler = upscaler ?? Renderer.LumaUpscaler;
                m_Downscaler = downscaler ?? Renderer.LumaDownscaler;
                m_Convolver = convolver;
            }

            #region Implementation

            private TextureSize m_InputSize;
            private readonly TextureSize m_OutputSize;
            private readonly TextureChannels m_Channels;
            private readonly Vector2 m_Offset;
            private readonly TextureFormat? m_Format;

            private readonly IScaler m_Upscaler;
            private readonly IScaler m_Downscaler;
            private readonly IScaler m_Convolver;

            public string Description()
            {
                return StatusHelpers.ScaleDescription(m_InputSize, m_OutputSize, m_Upscaler, m_Downscaler, m_Convolver);
            }

            public Process ReSize(TextureSize targetSize)
            {
                return new Process(targetSize, m_Channels, m_Offset, m_Upscaler, m_Downscaler, m_Convolver, m_Format);
            }

            public Process ForceOffsetCorrection()
            {
                return new Process(m_OutputSize, m_Channels, m_Offset, m_Upscaler, m_Downscaler, m_Convolver ?? m_Upscaler, m_Format);
            }

            #endregion

            #region TextureProcess Implementation

            protected override TextureFormat CalcFormat(TextureFormat inputFormat)
            {
                return m_Format ?? inputFormat;
            }

            protected override TextureSize CalcSize(TextureSize inputSize)
            {
                m_InputSize = inputSize;
                return m_OutputSize;
            }

            protected override void Render(ITexture2D input, ITargetTexture target)
            {
                Renderer.Scale(target, input, m_Channels, m_Offset, m_Upscaler, m_Downscaler, m_Convolver);
            }

            #endregion
        }

        #endregion

        #region Implementation

        private Process m_Process;
        private bool m_Tagged;

        public string ScaleDescription
        {
            get { return m_Process.Description(); }
        }

        public void EnableTag()
        {
            if (m_Tagged)
                return;

            m_Tagged = true;
            this.AddLabel(ScaleDescription);
        }

        private ITextureFilter ReDo(Process process)
        {
            var result = new ResizeFilter(Undo(), process);
            if (m_Tagged)
                result.EnableTag();
            return result;
        }

        public ITextureFilter ForceOffsetCorrection()
        {
            return ReDo(m_Process.ForceOffsetCorrection());
        }

        public ITextureFilter ResizeTo(TextureSize targetSize)
        {
            return ReDo(m_Process.ReSize(targetSize)); ;
        }

        #endregion
    }

    public static class MergeProcesses
    {
        public static IShaderHandle MergeY_UV = GetShader("MergeY_UV.hlsl").GetHandle();
        public static IShaderHandle MergeY_U_V = GetShader("MergeY_U_V.hlsl").GetHandle();

        private static IShaderConfig GetShader(string shaderFile)
        {
            var shaderDataFilePath = Path.Combine(ShaderCache.ShaderPathRoot, "Common");

            return new Shader(DefinitionHelper.FromFile(Path.Combine(shaderDataFilePath, shaderFile)));
        }
    }
}
