using System;
using System.Drawing;
using System.IO;
using TransformFunc = System.Func<System.Drawing.Size, System.Drawing.Size>;

namespace Mpdn.RenderScript
{
    public interface IRenderChain
    {
        IFilter CreateFilter(IResizeableFilter sourceFilter);
    }

    public abstract class RenderChain : IRenderChain
    {
        public abstract IFilter CreateFilter(IResizeableFilter sourceFilter);

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
            return ShaderCache<IShader>.CompileShader(Path.Combine(ShaderDataFilePath, shaderFileName),
                s => Renderer.CompileShader(s));
        }

        protected IShader LoadShader(string shaderFileName)
        {
            return Renderer.LoadShader(shaderFileName);
        }

        protected IShader11 CompileShader11(string shaderFileName, string profile)
        {
            return CompileShader11(shaderFileName, "main", profile);
        }

        protected IShader11 CompileShader11(string shaderFileName, string entryPoint, string profile)
        {
            return ShaderCache<IShader11>.CompileShader(Path.Combine(ShaderDataFilePath, shaderFileName),
                s => Renderer.CompileShader11(s, entryPoint, profile));
        }

        protected IShader11 LoadShader11(string shaderFileName)
        {
            return Renderer.LoadShader11(shaderFileName);
        }

        #endregion
    }

    public class StaticChain : IRenderChain
    {
        private readonly Func<IResizeableFilter, IFilter> m_Compiler;

        public StaticChain(Func<IResizeableFilter, IFilter> compiler)
        {
            m_Compiler = compiler;
        }

        public IFilter CreateFilter(IResizeableFilter sourceFilter)
        {
            return m_Compiler(sourceFilter);
        }
    }

    public class FilterChain
    {
        public IResizeableFilter Filter;

        public FilterChain(IResizeableFilter sourceFilter)
        {
            Filter = sourceFilter;
        }

        public Size OutputSize
        {
            get { return Filter.OutputSize; }
        }

        public void Add(IRenderChain renderChain)
        {
            Filter = renderChain.CreateFilter(Filter).MakeResizeable();
        }
    }

    public abstract class CombinedChain : RenderChain
    {
        protected abstract void BuildChain(FilterChain chain);

        public sealed override IFilter CreateFilter(IResizeableFilter sourceFilter)
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