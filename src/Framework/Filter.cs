using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using Mpdn.RenderScript.Scaler;
using SharpDX;
using TransformFunc = System.Func<System.Drawing.Size, System.Drawing.Size>;

namespace Mpdn.RenderScript
{
    public interface ITextureCache
    {
        ITexture GetTexture(Size textureSize);
        void PutTexture(ITexture texture);
        void PutTempTexture(ITexture texture);
    }

    public interface IFilter
    {
        IFilter[] InputFilters { get; }
        ITexture OutputTexture { get; }
        Size OutputSize { get; }
        int FilterIndex { get; }
        int LastDependentIndex { get; }
        void NewFrame();
        void Render(ITextureCache cache);
        void ReleaseTexture(ITextureCache cache);
        void Initialize(int time = 1);
        IFilter ConvertToRgb();
        IFilter ConvertToYuv();
        IFilter ScaleTo(Size outputSize);
    }

    public class TextureCache : ITextureCache
    {
        private List<ITexture> m_OldTextures = new List<ITexture>();
        private List<ITexture> m_SavedTextures = new List<ITexture>();
        private List<ITexture> m_TempTextures = new List<ITexture>();

        public ITexture GetTexture(Size textureSize)
        {
            foreach (var list in new[] {m_SavedTextures, m_OldTextures})
            {
                var index = list.FindIndex(x => (x.Width == textureSize.Width) && (x.Height == textureSize.Height));
                if (index < 0) continue;

                var texture = list[index];
                list.RemoveAt(index);
                return texture;
            }

            return Renderer.CreateRenderTarget(textureSize);
        }

        public void PutTempTexture(ITexture texture)
        {
            m_TempTextures.Add(texture);
            m_SavedTextures.Add(texture);
        }

        public void PutTexture(ITexture texture)
        {
            m_SavedTextures.Add(texture);
        }

        public void FlushTextures()
        {
            foreach (var texture in m_OldTextures)
                Common.Dispose(texture);

            foreach (var texture in m_TempTextures)
                m_SavedTextures.Remove(texture);

            m_OldTextures = m_SavedTextures;
            m_TempTextures = new List<ITexture>();
            m_SavedTextures = new List<ITexture>();
        }

        public void Dispose()
        {
            FlushTextures();
            FlushTextures();
        }
    }

    public abstract class Filter : IFilter
    {
        protected Filter(params IFilter[] inputFilters)
        {
            if (inputFilters == null || inputFilters.Length == 0 || inputFilters.Any(f => f == null))
            {
                throw new ArgumentNullException("inputFilters");
            }

            Initialized = false;
            InputFilters = inputFilters;
        }

        public abstract void Render(IEnumerable<ITexture> inputs);

        #region IFilter Implementation

        protected bool Updated { get; set; }
        protected bool Initialized { get; set; }

        public IFilter[] InputFilters { get; private set; }
        public ITexture OutputTexture { get; private set; }

        public abstract Size OutputSize { get; }

        public int FilterIndex { get; private set; }
        public int LastDependentIndex { get; private set; }

        public virtual void Initialize(int time = 1)
        {
            LastDependentIndex = time;

            if (Initialized)
                return;

            foreach (var filter in InputFilters)
            {
                filter.Initialize(LastDependentIndex);
                LastDependentIndex = filter.LastDependentIndex;
            }

            FilterIndex = LastDependentIndex;

            foreach (var filter in InputFilters)
            {
                filter.Initialize(FilterIndex);
            }

            LastDependentIndex++;

            Initialized = true;
        }

        public virtual void NewFrame()
        {
            if (InputFilters == null)
                return;

            if (!Updated)
                return;

            Updated = false;

            foreach (var filter in InputFilters)
            {
                filter.NewFrame();
            }
        }

        public virtual void Render(ITextureCache cache)
        {
            if (Updated)
                return;

            Updated = true;

            foreach (var filter in InputFilters)
                filter.Render(cache);

            var inputTextures = InputFilters.Select(f => f.OutputTexture);

            OutputTexture = cache.GetTexture(OutputSize);

            Render(inputTextures);

            foreach (var filter in InputFilters)
            {
                if (filter.LastDependentIndex <= FilterIndex)
                    filter.ReleaseTexture(cache);
            }
        }

        public virtual void ReleaseTexture(ITextureCache cache)
        {
            if (OutputTexture == null)
                return;

            cache.PutTexture(OutputTexture);
            OutputTexture = null;
        }

        #endregion

        #region Conversions

        public virtual IFilter ConvertToRgb()
        {
            return new RgbFilter(this);
        }

        public virtual IFilter ConvertToYuv()
        {
            return new YuvFilter(this);
        }

        public virtual IFilter ScaleTo(Size outputSize)
        {
            return new ResizeFilter(this, outputSize, Renderer.LumaUpscaler, Renderer.LumaDownscaler);
        }

        #endregion
    }

    public abstract class BaseSourceFilter : IFilter
    {
        protected BaseSourceFilter(params IFilter[] inputFilters)
        {
            InputFilters = inputFilters;
        }

        #region IFilter Implementation

        public IFilter[] InputFilters { get; protected set; }
        public abstract ITexture OutputTexture { get; }

        public abstract Size OutputSize { get; }

        public virtual int FilterIndex
        {
            get { return 0; }
        }

        public virtual int LastDependentIndex { get; private set; }

        public void Initialize(int time = 1)
        {
            LastDependentIndex = time;
        }

        public void NewFrame()
        {
        }

        public void Render(ITextureCache cache)
        {
        }

        public virtual void ReleaseTexture(ITextureCache cache)
        {
            cache.PutTempTexture(OutputTexture);
        }

        public void Dispose()
        {
        }

        #endregion

        #region Conversions

        public virtual IFilter ConvertToRgb()
        {
            return new RgbFilter(this);
        }

        public virtual IFilter ConvertToYuv()
        {
            return new YuvFilter(this);
        }

        public virtual IFilter ScaleTo(Size outputSize)
        {
            return new ResizeFilter(this, outputSize, Renderer.LumaUpscaler, Renderer.LumaDownscaler);
        }

        #endregion
    }

    public sealed class SourceFilter : BaseSourceFilter
    {
        public bool WantYuv { get; private set; }

        #region IFilter Implementation

        public override ITexture OutputTexture
        {
            get { return Renderer.InputRenderTarget; }
        }

        public override Size OutputSize
        {
            get
            {
                return GetOutputSize();
            }
        }

        private Size m_OutputSize;

        public Size GetOutputSize(bool assumeVideoSizeIfEmpty = true)
        {
            return assumeVideoSizeIfEmpty ? (m_OutputSize.IsEmpty ? Renderer.VideoSize : m_OutputSize) : m_OutputSize;
        }

        public override IFilter ConvertToRgb()
        {
            WantYuv = false;
            return this;
        }

        public override IFilter ConvertToYuv()
        {
            WantYuv = true;
            return this;
        }

        public override IFilter ScaleTo(Size outputSize)
        {
            m_OutputSize = outputSize;
            return this;
        }

        #endregion
    }

    public class YSourceFilter : BaseSourceFilter
    {
        public override ITexture OutputTexture
        {
            get { return Renderer.TextureY; }
        }

        public override Size OutputSize
        {
            get { return Renderer.LumaSize; }
        }

        public override void ReleaseTexture(ITextureCache cache)
        {
        }
    }

    public class USourceFilter : BaseSourceFilter
    {
        public override ITexture OutputTexture
        {
            get { return Renderer.TextureU; }
        }

        public override Size OutputSize
        {
            get { return Renderer.ChromaSize; }
        }

        public override void ReleaseTexture(ITextureCache cache)
        {
        }
    }

    public class VSourceFilter : BaseSourceFilter
    {
        public override ITexture OutputTexture
        {
            get { return Renderer.TextureV; }
        }

        public override Size OutputSize
        {
            get { return Renderer.ChromaSize; }
        }

        public override void ReleaseTexture(ITextureCache cache)
        {
        }
    }

    public class RgbFilter : Filter
    {
        public RgbFilter(IFilter inputFilter)
            : base(inputFilter)
        {
        }

        public override Size OutputSize
        {
            get { return InputFilters[0].OutputSize; }
        }

        public override void Render(IEnumerable<ITexture> inputs)
        {
            Renderer.ConvertToRgb(OutputTexture, inputs.Single(), Renderer.Colorimetric);
        }

        public override IFilter ConvertToYuv()
        {
            return InputFilters[0];
        }
    }

    public class YuvFilter : Filter
    {
        public YuvFilter(IFilter inputFilter)
            : base(inputFilter)
        {
        }

        public override Size OutputSize
        {
            get { return InputFilters[0].OutputSize; }
        }

        public override void Render(IEnumerable<ITexture> inputs)
        {
            Renderer.ConvertToYuv(OutputTexture, inputs.Single(), Renderer.Colorimetric);
        }

        public override IFilter ConvertToRgb()
        {
            return InputFilters[0];
        }
    }

    public class ResizeFilter : Filter
    {
        private readonly IScaler m_Downscaler;
        private readonly IScaler m_Upscaler;

        public ResizeFilter(IFilter inputFilter, Size outputSize)
            : this(inputFilter, outputSize, Renderer.LumaUpscaler, Renderer.LumaDownscaler)
        { }

        public ResizeFilter(IFilter inputFilter, Size outputSize, IScaler upscaler, IScaler downscaler)
            : base(inputFilter)
        {
            m_OutputSize = outputSize;
            m_Upscaler = upscaler;
            m_Downscaler = downscaler;
        }

        private Size m_OutputSize;

        public override Size OutputSize
        {
            get { return m_OutputSize; }
        }

        public override void Render(IEnumerable<ITexture> inputs)
        {
            Renderer.Scale(OutputTexture, inputs.Single(), m_Upscaler, m_Downscaler);
        }

        public override IFilter ScaleTo(Size outputSize)
        {
            m_OutputSize = outputSize;
            return this;
        }
    }

    public class ShaderFilter : Filter
    {
        public ShaderFilter(IShader shader, TransformFunc transform, int sizeIndex, bool linearSampling, float[] arguments,
            params IFilter[] inputFilters)
            : base(inputFilters)
        {
            if (sizeIndex < 0 || sizeIndex >= inputFilters.Length || inputFilters[sizeIndex] == null)
            {
                throw new IndexOutOfRangeException(String.Format("No valid input filter at index {0}", sizeIndex));
            }

            Shader = shader;
            LinearSampling = linearSampling;
            Transform = transform;
            SizeIndex = sizeIndex;

            arguments = arguments ?? new float[0];
            args = new float[4*((arguments.Length + 3) / 4)];
            arguments.CopyTo(args, 0);
        }

        protected IShader Shader { get; private set; }
        protected bool LinearSampling { get; private set; }
        protected int Counter { get; private set; }
        protected TransformFunc Transform { get; private set; }
        protected int SizeIndex { get; private set; }
        protected float[] args;

        public override Size OutputSize
        {
            get { return Transform(InputFilters[SizeIndex].OutputSize); }
        }

        public override void Render(IEnumerable<ITexture> inputs)
        {
            LoadInputs(inputs);
            Renderer.Render(OutputTexture, Shader);
        }

        protected virtual void LoadInputs(IEnumerable<ITexture> inputs)
        {
            var i = 0;
            foreach (var input in inputs)
            {
                Shader.SetTextureConstant(i, input, LinearSampling, false);
                Shader.SetConstant(String.Format("size{0}", i),
                    new Vector4(input.Width, input.Height, 1.0f/input.Width, 1.0f/input.Height), false);
                i++;
            }

            for (i = 0; 4*i < args.Length; i++)
                Shader.SetConstant(String.Format("args{0}", i), new Vector4(args[4*i], args[4*i+1], args[4*i+2], args[4*i+3]), false);

            // Legacy constants 
            var output = OutputTexture;
            Shader.SetConstant(0, new Vector4(output.Width, output.Height, Counter++, Stopwatch.GetTimestamp()),
                false);
            Shader.SetConstant(1, new Vector4(1.0f/output.Width, 1.0f/output.Height, 0, 0), false);
        }

        #region Auxilary Constructors

        public ShaderFilter(IShader shader, params IFilter[] inputFilters)
            : this(shader, false, inputFilters)
        {
        }

        public ShaderFilter(IShader shader, bool linearSampling, params IFilter[] inputFilters)
            : this(shader, 0, linearSampling, inputFilters)
        {
        }

        public ShaderFilter(IShader shader, int sizeIndex, params IFilter[] inputFilters)
            : this(shader, sizeIndex, false, inputFilters)
        {
        }

        public ShaderFilter(IShader shader, int sizeIndex, bool linearSampling, params IFilter[] inputFilters)
            : this(shader, s => s, sizeIndex, linearSampling, new float[0], inputFilters)
        {
        }

        public ShaderFilter(IShader shader, TransformFunc transform, params IFilter[] inputFilters)
            : this(shader, transform, 0, false, new float[0], inputFilters)
        {
        }

        public ShaderFilter(IShader shader, TransformFunc transform, bool linearSampling, params IFilter[] inputFilters)
            : this(shader, transform, 0, linearSampling, new float[0], inputFilters)
        {
        }

        public ShaderFilter(IShader shader, TransformFunc transform, int sizeIndex, params IFilter[] inputFilters)
            : this(shader, transform, sizeIndex, false, new float[0], inputFilters)
        {
        }

        public ShaderFilter(IShader shader, float[] arguments, params IFilter[] inputFilters)
            : this(shader, false, arguments, inputFilters)
        {
        }

        public ShaderFilter(IShader shader, bool linearSampling, float[] arguments, params IFilter[] inputFilters)
            : this(shader, 0, linearSampling, arguments, inputFilters)
        {
        }

        public ShaderFilter(IShader shader, int sizeIndex, float[] arguments, params IFilter[] inputFilters)
            : this(shader, sizeIndex, false, arguments, inputFilters)
        {
        }

        public ShaderFilter(IShader shader, int sizeIndex, bool linearSampling, float[] arguments, params IFilter[] inputFilters)
            : this(shader, s => s, sizeIndex, linearSampling, arguments, inputFilters)
        {
        }

        public ShaderFilter(IShader shader, TransformFunc transform, float[] arguments, params IFilter[] inputFilters)
            : this(shader, transform, 0, false, arguments, inputFilters)
        {
        }

        public ShaderFilter(IShader shader, TransformFunc transform, bool linearSampling, float[] arguments, params IFilter[] inputFilters)
            : this(shader, transform, 0, linearSampling, arguments, inputFilters)
        {
        }

        public ShaderFilter(IShader shader, TransformFunc transform, int sizeIndex, float[] arguments, params IFilter[] inputFilters)
            : this(shader, transform, sizeIndex, false, arguments, inputFilters)
        {
        }

        #endregion
    }
}