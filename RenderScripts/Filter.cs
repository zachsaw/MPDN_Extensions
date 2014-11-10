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
        bool TextureStolen { get; set; }
        void NewFrame();
        void Render();
        void Initialize(int time = 0);
        void AllocateTextures();
        void DeallocateTextures();
    }

    public abstract class Filter : IFilter
    {
        private bool m_Disposed;
        private ITexture m_OutputTexture;
        private bool m_OwnsTexture;

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

        protected IRenderer Renderer { get; private set; }
        protected bool Updated { get; set; }
        protected bool Initialized { get; set; }

        #region IFilter

        public void Dispose()
        {
            Dispose(true);
        }

        public IFilter[] InputFilters { get; private set; }

        public ITexture OutputTexture
        {
            get
            {
                // If m_OutputTexture is null, we are reusing  
                // Renderer.OutputRenderTarget as our OutputTexture 
                return m_OutputTexture ?? Renderer.OutputRenderTarget;
            }
            private set { m_OutputTexture = value; }
        }

        public abstract Size OutputSize { get; }

        public int FilterIndex { get; private set; }
        public int LastDependentIndex { get; private set; }
        public bool TextureStolen { get; set; }

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

        public virtual void AllocateTextures()
        {
            var size = OutputSize;
            if (OutputTexture == null || size.Width != OutputTexture.Width || size.Height != OutputTexture.Height)
            {
                if (m_OwnsTexture)
                {
                    Common.Dispose(OutputTexture);
                }
                OutputTexture = null;
            }

            foreach (var filter in InputFilters)
            {
                filter.AllocateTextures();
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

            if (m_OwnsTexture)
            {
                Common.Dispose(OutputTexture);
            }
            OutputTexture = null;
        }

        #endregion

        public abstract void Render(IEnumerable<ITexture> inputs);

        private void StealTexture(IFilter filter) {
            OutputTexture = filter.OutputTexture;
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
                        f.OutputTexture != null &&
                        f.OutputTexture.Width == OutputSize.Width &&
                        f.OutputTexture.Height == OutputSize.Height);

            if (filter != null)
            {
                // Steal texture
                StealTexture(filter);
                return;
            }

            // Doesn't work yet, added workaround in SourceFilter.
            /*if (OutputSize == Renderer.OutputSize)
            {
                // Reuse renderer's output render target as texture
                OutputTexture = null;
                return;
            }*/

            OutputTexture = Renderer.CreateRenderTarget(OutputSize);
            m_OwnsTexture = true;
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

            if (!m_OwnsTexture)
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

    public class SourceFilter : IFilter
    {
        public SourceFilter(IRenderer renderer)
        {
            Renderer = renderer;
            InputFilters = new[]{new OutputDummy(renderer)};
        }

        protected IRenderer Renderer { get; private set; }

        public IFilter[] InputFilters { get; private set; }

        public virtual ITexture OutputTexture
        {
            get { return Renderer.InputRenderTarget; }
        }

        public virtual Size OutputSize
        {
            get { return Renderer.InputSize; }
        }

        public int FilterIndex
        {
            get { return -1; }
        }

        public int LastDependentIndex { get; private set; }

        public bool TextureStolen { get; set; }

        public void Dispose()
        {
        }

        public void NewFrame()
        {
        }

        public void Render()
        {
        }

        public void Initialize(int time = 0)
        {
            LastDependentIndex = time;
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
    }

    public class OutputDummy : IFilter {
        public OutputDummy(IRenderer renderer) {
            Renderer = renderer;
            InputFilters = null;
        }

        protected IRenderer Renderer { get; private set; }

        public IFilter[] InputFilters { get; private set; }

        public ITexture OutputTexture
        {
            get { return Renderer.OutputRenderTarget; }
        }

        public Size OutputSize
        {
            get { return Renderer.TargetSize; }
        }

        public int FilterIndex
        {
            get { return -2; }
        }

        public int LastDependentIndex 
        { 
            get { return -1; }
        }

        public void Initialize(int time = 0) { }
        public bool TextureStolen { get; set; }
        public void Dispose() { }
        public void NewFrame() { }
        public void Render() { }

        public void AllocateTextures() { 
            TextureStolen = false; 
        }

        public void DeallocateTextures() { }
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

        private void LoadInputs(IEnumerable<ITexture> inputs)
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
