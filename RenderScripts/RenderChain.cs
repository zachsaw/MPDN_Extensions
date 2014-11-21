using System;
using Mpdn.RenderScript.Scaler;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.IO;
using SharpDX;
using TransformFunc = System.Func<System.Drawing.Size, System.Drawing.Size>;
using YAXLib;

namespace Mpdn.RenderScript
{
    public interface IRenderChain
    {
        IFilter CreateFilter(IFilter sourceFilter);
        bool WantYuv { get; }
        Size PrescaleSize { get;  }
    }

    public abstract class RenderChain : IRenderChain
    {
        public abstract IFilter CreateFilter(IFilter sourceFilter);
        [YAXDontSerialize]
        public virtual bool WantYuv { get; protected set; }
        [YAXDontSerialize]
        public virtual Size PrescaleSize { get; protected set; }

        #region Convenience Functions

        protected virtual string ShaderPath
        {
            get { return GetType().Name; }
        }

        protected string ShaderDataFilePath
        {
            get
            {
                var asmPath = typeof(IRenderScript).Assembly.Location;
                return Path.Combine(Common.GetDirectoryName(asmPath), "RenderScripts", ShaderPath);
            }
        }

        protected IShader CompileShader(string shaderFileName)
        {
            return ShaderCache.CompileShader(Path.Combine(ShaderDataFilePath, shaderFileName));
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

            return new ShaderFilter(shader, transform, sizeIndex, linearSampling, inputFilters);
        }

        #endregion
    }

    public class StaticChain : IRenderChain
    {
        private Func<IFilter, IFilter> Compiler;
        [YAXDontSerialize]
        public virtual bool WantYuv { get; protected set; }
        [YAXDontSerialize]
        public virtual Size PrescaleSize { get; protected set; }

        public StaticChain(Func<IFilter, IFilter> compiler)
        {
            Compiler = compiler;
        }

        public IFilter CreateFilter(IFilter sourceFilter)
        {
            return Compiler(sourceFilter);
        }
    }

    public class FilterChain {
        public IFilter Filter;

        public bool WantYuv { 
            get {
                if (first != null)
                    return first.WantYuv;
                else
                    return false;
            }
        }
        public Size PrescaleSize
        {
            get
            {
                if (first != null)
                    return first.PrescaleSize;
                else
                    return Size.Empty;
            }
        }

        private IRenderChain first;
        private IRenderChain previous;

        public FilterChain(IFilter sourceFilter)
        {
            Filter = sourceFilter;
        }

        public void Add(IRenderChain current)
        {
            if (previous != null)
            {
                // Convert to/from YUV
                if (previous.WantYuv && !current.WantYuv)
                    Filter = new RgbFilter(Filter);
                else if (!previous.WantYuv && current.WantYuv)
                    throw new NotImplementedException();
                //Filter = new YuvFilter(Filter);

                // Scale if necessary
                if (!current.PrescaleSize.IsEmpty && Filter.OutputSize != current.PrescaleSize)
                    Filter = new ResizeFilter(Filter, () => current.PrescaleSize, Renderer.LumaUpscaler, Renderer.LumaDownscaler);
            }
            else first = current;

            Filter = current.CreateFilter(Filter);
            previous = current;
        }

        public Size OutputSize
        {
            get { return Filter.OutputSize; }
        }
    }

    public abstract class CombinedChain : RenderChain
    {
        protected abstract void BuildChain(FilterChain Chain);

        public override IFilter CreateFilter(IFilter sourceFilter) {
            var chain = new FilterChain(sourceFilter);
            BuildChain(chain);

            WantYuv = chain.WantYuv;
            PrescaleSize = chain.PrescaleSize;
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