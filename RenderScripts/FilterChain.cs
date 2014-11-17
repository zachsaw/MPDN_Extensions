using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.IO;
using SharpDX;
using TransformFunc = System.Func<System.Drawing.Size, System.Drawing.Size>;

namespace Mpdn.RenderScript
{
    public abstract class FilterChain : IDisposable
    {
        public abstract Func<IFilter, IFilter> Compile();

        public abstract void Dispose();

        public FilterChain Append(FilterChain filterChain)
        {
            return new ComposedChain(this, filterChain);
        }

        private class ComposedChain : FilterChain {
            FilterChain A, B;

            public ComposedChain(FilterChain chainA, FilterChain chainB)
            {
                A = chainA;
                B = chainB;
            }

            public override Func<IFilter, IFilter> Compile()
            {
                var f = A.Compile();
                var g = B.Compile();
                return x => g(f(x));
            }

            public override void Dispose()
            {
                Common.Dispose(A);
                Common.Dispose(B);
            }
        }
    }

    public abstract class ChainBuilder<TSettings> : FilterChain
        where TSettings : class, new()
    {
        public IRenderer Renderer;
        public TSettings Settings { private get; set; }

        public ChainBuilder()
        {
            m_CompiledShaders = new Dictionary<string,IShader>();
            Settings = new TSettings();
        }

        protected abstract IFilter CreateFilter(IFilter sourceFilter, TSettings settings);

        public override Func<IFilter, IFilter> Compile()
        {
            return x => CreateFilter(x, Settings);
        }

        #region Convenience Functions

        private Dictionary<String, IShader> m_CompiledShaders;

        public override void Dispose()
        {
            foreach (var shader in m_CompiledShaders)
            {
                Common.Dispose(shader);
            }
        }

        protected virtual string ShaderPath
        {
            get { return GetType().FullName; }
        }

        protected string ShaderDataFilePath
        {
            get
            {
                var asmPath = typeof(IScriptRenderer).Assembly.Location;
                return Path.Combine(Common.GetDirectoryName(asmPath), "RenderScripts", ShaderPath);
            }
        }

        protected virtual void Scale(ITexture output, ITexture input)
        {
            Renderer.Scale(output, input, Renderer.LumaUpscaler, Renderer.LumaDownscaler);
        }

        protected IShader CompileShader(string shaderFileName)
        {
            IShader shader;
            m_CompiledShaders.TryGetValue(shaderFileName, out shader);

            if (shader == null)
            {
                shader = Renderer.CompileShader(Path.Combine(ShaderDataFilePath, shaderFileName));
                m_CompiledShaders.Add(shaderFileName, shader);
            }

            return shader;
        }

        protected IFilter CreateFilter(IShader shader, params IFilter[] inputFilters)
        {
            return CreateFilter(shader, false, inputFilters);
        }

        protected IFilter CreateFilter(IShader shader, bool linearSampling, params IFilter[] inputFilters)
        {
            return CreateFilter(shader, 0, linearSampling, inputFilters);
        }

        protected IFilter CreateFilter(IShader shader, int sizeIndex, params IFilter[] inputFilters)
        {
            return CreateFilter(shader, sizeIndex, false, inputFilters);
        }

        protected IFilter CreateFilter(IShader shader, int sizeIndex, bool linearSampling, params IFilter[] inputFilters)
        {
            return CreateFilter(shader, s => new Size(s.Width, s.Height), sizeIndex, linearSampling, inputFilters);
        }

        protected IFilter CreateFilter(IShader shader, TransformFunc transform,
            params IFilter[] inputFilters)
        {
            return CreateFilter(shader, transform, 0, false, inputFilters);
        }

        protected IFilter CreateFilter(IShader shader, TransformFunc transform, bool linearSampling,
            params IFilter[] inputFilters)
        {
            return CreateFilter(shader, transform, 0, linearSampling, inputFilters);
        }

        protected IFilter CreateFilter(IShader shader, TransformFunc transform, int sizeIndex,
            params IFilter[] inputFilters)
        {
            return CreateFilter(shader, transform, sizeIndex, false, inputFilters);
        }

        protected IFilter CreateFilter(IShader shader, TransformFunc transform, int sizeIndex,
            bool linearSampling, params IFilter[] inputFilters)
        {
            if (shader == null)
                throw new ArgumentNullException("shader");

            if (Renderer == null)
                throw new InvalidOperationException("CreateFilter is not available before Setup() is called");

            return new ShaderFilter(Renderer, shader, transform, sizeIndex, linearSampling, inputFilters);
        }

        #endregion
    }

    public class NoSettings { }

    public abstract class ChainBuilder : ChainBuilder<NoSettings> {
        protected override IFilter CreateFilter(IFilter sourceFilter, NoSettings settings)
        {
            return CreateFilter(sourceFilter);
        }

        protected abstract IFilter CreateFilter(IFilter sourceFilter);
    }
}