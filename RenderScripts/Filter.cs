using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using SharpDX;
using TransformFunc = System.Func<System.Drawing.Size, System.Drawing.Size>;

namespace Mpdn.RenderScript
{
    public interface IFilter : ICloneable, IDisposable
    {
        IFilter[] InputFilters { get; }
        ITexture OutputTexture { get; }
        bool IsTextureRenderTarget { get; }
        Size OutputSize { get; }
        int FilterIndex { get; }
        int LastDependentIndex { get; }
        bool TextureStolen { get; set; }
        void NewFrame();
        void Render();
        void Initialize(int time = 0);
        void AllocateTextures();
        void DeallocateTextures();
        IFilter Append(IFilter filter);
    }

    public abstract class Filter : IFilter
    {
        private bool m_Disposed;
        private ITexture m_OutputTexture;
        private IFilter m_TextureOwner;

        protected Filter(IRenderer renderer, params IFilter[] inputFilters)
        {
            if (inputFilters == null || inputFilters.Length == 0 || inputFilters.Any(f => f == null))
            {
                throw new ArgumentNullException("inputFilters");
            }

            Renderer = renderer;
            Initialized = false;

            InputFilters = inputFilters;
        }

        private bool OwnsTexture
        {
            get
            {
                // null means we own this texture
                return m_TextureOwner == null;
            }
        }

        protected IRenderer Renderer { get; private set; }
        protected bool Updated { get; set; }
        protected bool Initialized { get; set; }

        #region IFilter Implementation

        public IFilter[] InputFilters { get; private set; }

        public ITexture OutputTexture
        {
            get
            {
                // If m_TextureOwner is not null then we are reusing the texture of m_TextureOwner
                return m_TextureOwner != null ? m_TextureOwner.OutputTexture : m_OutputTexture;
            }
            private set
            {
                m_OutputTexture = value;
                m_TextureOwner = null; // we own this texture
            }
        }

        public virtual bool IsTextureRenderTarget
        {
            get { return true; }
        }

        public abstract Size OutputSize { get; }

        public int FilterIndex { get; private set; }
        public int LastDependentIndex { get; private set; }
        public bool TextureStolen { get; set; }

        public void Dispose()
        {
            Dispose(true);
        }

        public virtual void Initialize(int time = 0)
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

        public virtual void Render()
        {
            if (Updated)
                return;

            Updated = true;

            foreach (var filter in InputFilters)
            {
                filter.Render();
            }

            var inputTextures = InputFilters.Select(f => f.OutputTexture);
            Render(inputTextures);
        }

        public virtual void AllocateTextures()
        {
            foreach (var filter in InputFilters)
            {
                filter.AllocateTextures();
            }

            var size = OutputSize;
            if (OutputTexture == null || size.Width != OutputTexture.Width || size.Height != OutputTexture.Height)
            {
                if (OwnsTexture)
                {
                    Common.Dispose(OutputTexture);
                }
                OutputTexture = null;
            }

            if (OutputTexture == null)
            {
                AllocateOutputTexture();
            }
        }

        public virtual void DeallocateTextures()
        {
            foreach (var filter in InputFilters)
            {
                filter.DeallocateTextures();
            }

            if (OwnsTexture)
            {
                Common.Dispose(OutputTexture);
            }
            OutputTexture = null;
        }

        public IFilter Append(IFilter filter)
        {
            return filter as SourceFilter != null ? this : AppendFilter(filter);
        }

        private IFilter AppendFilter(IFilter filter)
        {
            // Clone the filters to make sure we aren't modifying the original filters
            filter = DeepCloneFilter(filter);
            var replacementFilter = DeepCloneFilter(this);
            // Replace source filters of 'filter' with 'this'
            return ReplaceSource(filter, replacementFilter);
        }

        private static IFilter ReplaceSource(IFilter filter, IFilter replacementFilter)
        {
            if (filter is SourceFilter)
                return replacementFilter;

            for (var i = 0; i < filter.InputFilters.Length; i++)
            {
                filter.InputFilters[i] = ReplaceSource(filter.InputFilters[i], replacementFilter);
            }

            return filter;
        }

        private static IFilter DeepCloneFilter(IFilter filter)
        {
            if (filter == null)
                return null;

            var f = (ICloneable) filter;
            var result = (IFilter) f.Clone();
            if (filter.InputFilters == null)
                return result;

            for (var i = 0; i < filter.InputFilters.Length; i++)
            {
                result.InputFilters[i] = DeepCloneFilter(result.InputFilters[i]);
            }
            return result;
        }

        #endregion

        public object Clone()
        {
            var clone = (Filter) MemberwiseClone();
            clone.InputFilters = (IFilter[]) InputFilters.Clone();
            return clone;
        }

        public abstract void Render(IEnumerable<ITexture> inputs);

        private void StealTexture(IFilter filter)
        {
            m_TextureOwner = filter;
            filter.TextureStolen = true;
        }

        private void AllocateOutputTexture()
        {
            TextureStolen = false;

            // This can (should) be improved - it currently does not reuse all feasible textures
            var filter =
                GetInputFilters(this)
                    .FirstOrDefault(f =>
                        !f.TextureStolen &&
                        f.LastDependentIndex < FilterIndex &&
                        f.IsTextureRenderTarget &&
                        f.OutputTexture != null &&
                        f.OutputTexture.Width == OutputSize.Width &&
                        f.OutputTexture.Height == OutputSize.Height);

            if (filter != null)
            {
                StealTexture(filter);
                return;
            }

            if (!OutputSize.IsEmpty)
                OutputTexture = Renderer.CreateRenderTarget(OutputSize);
        }

        private static IEnumerable<IFilter> GetInputFilters(IFilter filter)
        {
            var filters = filter.InputFilters;
            if (filters == null)
                return null;

            var result = new List<IFilter>();
            result.AddRange(filters.Where(f => f.LastDependentIndex <= filter.FilterIndex));

            foreach (var f in filters)
            {
                var inputFilters = GetInputFilters(f);
                if (inputFilters == null)
                    continue;

                result.AddRange(inputFilters);
            }

            return result;
        }

        ~Filter()
        {
            Dispose(false);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (m_Disposed)
                return;

            DisposeInputFilters();

            if (!OwnsTexture)
                return;

            Common.Dispose(OutputTexture);
            OutputTexture = null;

            m_Disposed = true;
        }

        private void DisposeInputFilters()
        {
            if (InputFilters == null)
                return;

            foreach (var filter in InputFilters)
            {
                Common.Dispose(filter);
            }
        }
    }

    public abstract class BaseSourceFilter : IFilter
    {
        protected BaseSourceFilter(IRenderer renderer, params IFilter[] inputFilters)
        {
            Renderer = renderer;
            InputFilters = inputFilters;
        }

        protected IRenderer Renderer { get; private set; }

        #region IFilter Implementation

        public IFilter[] InputFilters { get; protected set; }
        public abstract ITexture OutputTexture { get; }

        public virtual bool IsTextureRenderTarget
        {
            get { return false; }
        }

        public abstract Size OutputSize { get; }

        public int FilterIndex
        {
            get { return -1; }
        }

        public int LastDependentIndex { get; private set; }

        public bool TextureStolen { get; set; }

        public void Dispose()
        {
        }

        public void Initialize(int time = 0)
        {
            LastDependentIndex = time;
        }

        public void NewFrame()
        {
        }

        public void Render()
        {
        }

        public void AllocateTextures()
        {
            foreach (var filter in InputFilters)
            {
                filter.AllocateTextures();
            }
            TextureStolen = false;
        }

        public void DeallocateTextures()
        {
        }

        public IFilter Append(IFilter filter)
        {
            return filter;
        }

        public object Clone()
        {
            var clone = (BaseSourceFilter) MemberwiseClone();
            clone.InputFilters = (IFilter[]) InputFilters.Clone();
            return clone;
        }

        #endregion
    }

    public abstract class MetaFilter : IFilter
    {
        protected abstract IFilter Filter { get; }

        #region m_Filter Passthough

        public virtual void Dispose()
        {
            Filter.Dispose();
        }

        public virtual Object Clone()
        {
            return new ProxyFilter((IFilter)Filter.Clone());
        }

        public virtual IFilter[] InputFilters
        {
            get { return Filter.InputFilters; }
        }

        public virtual ITexture OutputTexture
        {
            get { return Filter.OutputTexture; }
        }

        public virtual bool IsTextureRenderTarget
        {
            get { return Filter.IsTextureRenderTarget; }
        }

        public virtual Size OutputSize
        {
            get { return Filter.OutputSize; }
        }

        public virtual int FilterIndex
        {
            get { return Filter.FilterIndex; }
        }

        public virtual int LastDependentIndex
        {
            get { return Filter.LastDependentIndex; }
        }

        public virtual bool TextureStolen
        {
            get { return Filter.TextureStolen; }
            set { Filter.TextureStolen = value; }
        }

        public virtual void NewFrame()
        {
            Filter.NewFrame();
        }

        public virtual void Render()
        {
            Filter.Render();
        }

        public virtual void Initialize(int time = 0)
        {
            Filter.Initialize(time);
        }

        public virtual void AllocateTextures()
        {
            Filter.AllocateTextures();
        }

        public virtual void DeallocateTextures()
        {
            Filter.DeallocateTextures();
        }

        public virtual IFilter Append(IFilter filter)
        {
            return Filter.Append(filter);
        }

        #endregion
    }

    public class ProxyFilter : MetaFilter
    {
        private IFilter m_Filter;
        protected override IFilter Filter { get { return m_Filter; } }

        public ProxyFilter(IFilter filter)
        {
            m_Filter = filter;
        }

        public void ReplaceWith(IFilter filter)
        {
            m_Filter = filter;
        }
    }

    public sealed class SourceFilter : BaseSourceFilter
    {
        public SourceFilter(IRenderer renderer)
            : base(renderer, new OutputDummy(renderer))
        {
        }

        #region IFilter Implementation

        public override ITexture OutputTexture
        {
            get { return Renderer.InputRenderTarget; }
        }

        public override bool IsTextureRenderTarget
        {
            get { return true; }
        }

        public override Size OutputSize
        {
            get { return Renderer.InputSize; }
        }

        #endregion

        private sealed class OutputDummy : IFilter
        {
            public OutputDummy(IRenderer renderer)
            {
                Renderer = renderer;
                InputFilters = null;
            }

            private IRenderer Renderer { get; set; }

            #region IFilter Implementation

            public IFilter[] InputFilters { get; private set; }

            public ITexture OutputTexture
            {
                get { return Renderer.OutputRenderTarget; }
            }

            public bool IsTextureRenderTarget
            {
                get { return true; }
            }

            public Size OutputSize
            {
                get { return Renderer.OutputSize; }
            }

            public int FilterIndex
            {
                get { return -2; }
            }

            public int LastDependentIndex
            {
                get { return -1; }
            }

            public bool TextureStolen { get; set; }

            public void Dispose()
            {
            }

            public void Initialize(int time = 0)
            {
            }

            public void NewFrame()
            {
            }

            public void Render()
            {
            }

            public void AllocateTextures()
            {
                TextureStolen = false;
            }

            public void DeallocateTextures()
            {
            }

            public IFilter Append(IFilter filter)
            {
                throw new InvalidOperationException();
            }

            public object Clone()
            {
                return MemberwiseClone();
            }

            #endregion
        }
    }

    public sealed class OutputFilter : MetaFilter
    {
        private IRenderer Renderer;
        private IFilter m_Filter;
        protected override IFilter Filter { get { return m_Filter; } }

        public OutputFilter(IRenderer renderer, IFilter filter)
        {
            Renderer = renderer;
            m_Filter = filter;
            Initialize();
            AllocateTextures();
        }

        public override void Render()
        {
            Filter.NewFrame();
            Filter.Render();
            var output = Filter.OutputTexture;
            Scale(Renderer.OutputRenderTarget, output);
        }

        private void Scale(ITexture output, ITexture input)
        {
            Renderer.Scale(output, input, Renderer.LumaUpscaler, Renderer.LumaDownscaler);
        }
    }

    public class YSourceFilter : BaseSourceFilter
    {
        public YSourceFilter(IRenderer renderer) : base(renderer)
        {
        }

        public override ITexture OutputTexture
        {
            get { return Renderer.TextureY; }
        }

        public override Size OutputSize
        {
            get { return Renderer.LumaSize; }
        }
    }

    public class USourceFilter : BaseSourceFilter
    {
        public USourceFilter(IRenderer renderer) : base(renderer)
        {
        }

        public override ITexture OutputTexture
        {
            get { return Renderer.TextureU; }
        }

        public override Size OutputSize
        {
            get { return Renderer.ChromaSize; }
        }
    }

    public class VSourceFilter : BaseSourceFilter
    {
        public VSourceFilter(IRenderer renderer) : base(renderer)
        {
        }

        public override ITexture OutputTexture
        {
            get { return Renderer.TextureV; }
        }

        public override Size OutputSize
        {
            get { return Renderer.ChromaSize; }
        }
    }

    public class RgbFilter : Filter
    {
        public RgbFilter(IRenderer renderer, IFilter inputFilter)
            : base(renderer, inputFilter)
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
    }

    public class ShaderFilter : Filter
    {
        public ShaderFilter(IRenderer renderer, IShader shader, int sizeIndex, bool linearSampling,
            params IFilter[] inputFilters)
            : this(renderer, shader, s => new Size(s.Width, s.Height), sizeIndex, linearSampling, inputFilters)
        {
        }

        public ShaderFilter(IRenderer renderer, IShader shader, TransformFunc transform, int sizeIndex,
            bool linearSampling, params IFilter[] inputFilters)
            : base(renderer, inputFilters)
        {
            if (sizeIndex < 0 || sizeIndex >= inputFilters.Length || inputFilters[sizeIndex] == null)
            {
                throw new IndexOutOfRangeException(String.Format("No valid input filter at index {0}", sizeIndex));
            }

            Shader = shader;
            LinearSampling = linearSampling;
            Transform = transform;
            SizeIndex = sizeIndex;
        }

        protected IShader Shader { get; private set; }
        protected bool LinearSampling { get; private set; }
        protected int Counter { get; private set; }
        protected TransformFunc Transform { get; private set; }
        protected int SizeIndex { get; private set; }

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
                Shader.SetTextureConstant(i, input.Texture, LinearSampling, false);
                Shader.SetConstant(String.Format("size{0}", i),
                    new Vector4(input.Width, input.Height, 1.0f/input.Width, 1.0f/input.Height), false);
                i++;
            }

            // Legacy constants 
            var output = OutputTexture;
            Shader.SetConstant(0, new Vector4(output.Width, output.Height, Counter++, Stopwatch.GetTimestamp()),
                false);
            Shader.SetConstant(1, new Vector4(1.0f/output.Width, 1.0f/output.Height, 0, 0), false);
        }
    }
}