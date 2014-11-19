using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using SharpDX;
using TransformFunc = System.Func<System.Drawing.Size, System.Drawing.Size>;

namespace Mpdn.RenderScript
{
    public interface IFilter : IDisposable
    {
        IFilter[] InputFilters { get; }
        ITexture OutputTexture { get; }
        Size OutputSize { get; }
        int FilterIndex { get; }
        int LastDependentIndex { get; }
        void NewFrame();
        void Render(ITextureCache Cache);
        void ReleaseTexture(ITextureCache Cache);
        void Initialize(int time = 1);
    }

    public interface ITextureCache
    {
        ITexture GetTexture(Size textureSize);
        void     PutTexture(ITexture texture);
        void PutTempTexture(ITexture texture);
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

        public void Dispose()
        {
            Dispose(true);
        }

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

        public virtual void Render(ITextureCache Cache)
        {
            if (Updated)
                return;

            Updated = true;

            foreach (var filter in InputFilters)
                filter.Render(Cache);

            OutputTexture = Cache.GetTexture(OutputSize);

            var inputTextures = InputFilters.Select(f => f.OutputTexture);
            Render(inputTextures);

            foreach (var filter in InputFilters)
            {
                if (filter.LastDependentIndex == FilterIndex)
                    filter.ReleaseTexture(Cache);
            }
        }

        public virtual void ReleaseTexture(ITextureCache Cache) {
            Cache.PutTexture(OutputTexture);
            OutputTexture = null;
        }

        #endregion

        #region Destructors

        private bool m_Disposed;

        ~Filter()
        {
            Dispose(false);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (m_Disposed)
                return;

            DisposeInputFilters();

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

        #endregion
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

        public abstract Size OutputSize { get; }

        public virtual int FilterIndex
        {
            get { return 0; }
        }

        public virtual int LastDependentIndex { get; private set; }

        public void Dispose()
        {
        }

        public void Initialize(int time = 1)
        {
            LastDependentIndex = time;
        }

        public void NewFrame()
        {
        }

        public void Render(ITextureCache Cache)
        {
        }

        public virtual void ReleaseTexture(ITextureCache Cache)
        {
            Cache.PutTempTexture(OutputTexture);
        }

        #endregion
    }

    public sealed class SourceFilter : BaseSourceFilter
    {
        public SourceFilter(IRenderer renderer)
            : base(renderer)
        {
        }

        #region IFilter Implementation

        public override ITexture OutputTexture
        {
            get { return Renderer.InputRenderTarget; }
        }

        public override Size OutputSize
        {
            get { return Renderer.InputSize; }
        }

        #endregion
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

        public override void ReleaseTexture(ITextureCache Cache)
        {
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

        public override void ReleaseTexture(ITextureCache Cache)
        {
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

        public override void ReleaseTexture(ITextureCache Cache)
        {
        }
    }

    public class RgbFilter : Filter
    {
        public RgbFilter(IRenderer renderer, IFilter inputFilter)
            : base(inputFilter)
        {
        }

        public override Size OutputSize
        {
            get { return InputFilters[0].OutputSize; }
        }

        public override void Render(IEnumerable<ITexture> inputs)
        {
            StaticRenderer.ConvertToRgb(OutputTexture, inputs.Single());
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
        }

        protected IRenderer Renderer { get; private set; }
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
            StaticRenderer.Render(OutputTexture, Shader);
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

    public interface IOutputFilter
    {
        ScriptInterfaceDescriptor Descriptor { get; }
        void Refresh();
        void Render();
    }

    public class OutputFilter : IOutputFilter, IDisposable
    {
        protected IRenderer Renderer;
        protected IRenderChain Chain;
        private TextureCache Cache;
        private IFilter m_Filter;
        private SourceFilter m_SourceFilter;

        public OutputFilter(IRenderer renderer, IRenderChain chain)
        {
            Renderer = renderer;
            Chain = chain;
            m_SourceFilter = new SourceFilter(Renderer);
            Cache = new TextureCache(Renderer);
        }

        public ScriptInterfaceDescriptor Descriptor
        {
            get
            {
                return new ScriptInterfaceDescriptor
                {
                    OutputSize = m_Filter.OutputSize,
                    WantYuv = false,
                    Prescale = (m_SourceFilter.LastDependentIndex > 0)
                };
            }
        }

        public void Refresh()
        {
            Common.Dispose(m_Filter);

            m_Filter = Chain.CreateFilter(m_SourceFilter);
            m_Filter.Initialize();
        }

        public void Render()
        {
            Cache.PutTempTexture(Renderer.OutputRenderTarget);
            m_Filter.NewFrame();
            m_Filter.Render(Cache);
            Scale(Renderer.OutputRenderTarget, m_Filter.OutputTexture);
            m_Filter.ReleaseTexture(Cache);
            Cache.FlushTextures();
        }

        public void Dispose()
        {
            Common.Dispose(Cache);
            Common.Dispose(m_Filter);
            Common.Dispose(m_SourceFilter);
        }

        private void Scale(ITexture output, ITexture input)
        {
            Renderer.Scale(output, input, Renderer.LumaUpscaler, Renderer.LumaDownscaler);
        }

        #region TextureCache

        private class TextureCache : ITextureCache
        {
            private IRenderer Renderer;
            private List<ITexture> SavedTextures;
            private List<ITexture> OldTextures;
            private List<ITexture> TempTextures;

            public TextureCache(IRenderer renderer)
            {
                Renderer = renderer;
                SavedTextures = new List<ITexture>();
                OldTextures = new List<ITexture>();
                TempTextures = new List<ITexture>();
            }

            public ITexture GetTexture(Size textureSize)
            {
                foreach (var list in new[] { SavedTextures, OldTextures })
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
                TempTextures.Add(texture);
                SavedTextures.Add(texture);
            }

            public void PutTexture(ITexture texture)
            {
                SavedTextures.Add(texture);
            }

            public void FlushTextures()
            {
                foreach (var texture in OldTextures)
                    Common.Dispose(texture);

                foreach (var texture in TempTextures)
                    SavedTextures.Remove(texture);

                OldTextures = SavedTextures;
                TempTextures = new List<ITexture>();
                SavedTextures = new List<ITexture>();
            }

            public void Dispose()
            {
                FlushTextures();
                FlushTextures();
            }
        }

        #endregion
    }

}