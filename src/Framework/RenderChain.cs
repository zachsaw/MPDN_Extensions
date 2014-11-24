using System;
using System.Drawing;
using System.IO;
using TransformFunc = System.Func<System.Drawing.Size, System.Drawing.Size>;

namespace Mpdn.RenderScript
{
    public interface IRenderChain
    {
        IFilter CreateFilter(IFilter sourceFilter);
    }

    public abstract class RenderChain : IRenderChain
    {
        public abstract IFilter CreateFilter(IFilter sourceFilter);

        #region Shader Compilation

        protected virtual string ShaderPath
        {
            get { return GetType().Name; }
        }

        protected string ShaderDataFilePath
        {
            get
            {
                var asmPath = typeof (IRenderScript).Assembly.Location;
                return Path.Combine(Common.GetDirectoryName(asmPath), "RenderScripts", ShaderPath);
            }
        }

        protected IShader CompileShader(string shaderFileName)
        {
            return ShaderCache.CompileShader(Path.Combine(ShaderDataFilePath, shaderFileName));
        }

        #endregion
    }

    public class StaticChain : IRenderChain
    {
        private readonly Func<IFilter, IFilter> m_Compiler;

        public StaticChain(Func<IFilter, IFilter> compiler)
        {
            m_Compiler = compiler;
        }

        public IFilter CreateFilter(IFilter sourceFilter)
        {
            return m_Compiler(sourceFilter);
        }
    }

    public class FilterChain
    {
        public IFilter Filter;

        public FilterChain(IFilter sourceFilter)
        {
            Filter = sourceFilter;
        }

        public Size OutputSize
        {
            get { return Filter.OutputSize; }
        }

        public void Add(IRenderChain renderChain)
        {
            Filter = renderChain.CreateFilter(Filter);
        }
    }

    public abstract class CombinedChain : RenderChain
    {
        protected abstract void BuildChain(FilterChain chain);

        public override IFilter CreateFilter(IFilter sourceFilter)
        {
            var chain = new FilterChain(sourceFilter);
            BuildChain(chain);

            return chain.Filter;
        }

        #region Convenience functions

        protected bool IsDownscalingFrom(Size size)
        {
            return !IsNotScalingFrom(size) && !IsUpscalingFrom(size);
        }

        protected bool IsNotScalingFrom(Size size)
        {
            return size == Renderer.TargetSize;
        }

        protected bool IsUpscalingFrom(Size size)
        {
            var targetSize = Renderer.TargetSize;
            return targetSize.Width > size.Width || targetSize.Height > size.Height;
        }

        protected bool IsDownscalingFrom(FilterChain chain)
        {
            return IsDownscalingFrom(chain.OutputSize);
        }

        protected bool IsNotScalingFrom(FilterChain chain)
        {
            return IsNotScalingFrom(chain.OutputSize);
        }

        protected bool IsUpscalingFrom(FilterChain chain)
        {
            return IsUpscalingFrom(chain.OutputSize);
        }

        #endregion
    }
}