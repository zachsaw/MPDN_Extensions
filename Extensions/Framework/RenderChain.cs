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
// 
using System;
using System.IO;
using Mpdn.OpenCl;
using TransformFunc = System.Func<System.Drawing.Size, System.Drawing.Size>;

namespace Mpdn.RenderScript
{
    public abstract class RenderChain : IDisposable
    {
        public abstract IFilter CreateFilter(IResizeableFilter sourceFilter);

        #region Operators

        public static RenderChain Identity = new StaticChain(x => x);

        public static implicit operator Func<IFilter, IFilter>(RenderChain map)
        {
            return filter => map.CreateFilter(filter.MakeResizeable());
        }

        public static implicit operator RenderChain(Func<IResizeableFilter, IFilter> map)
        {
            return new StaticChain(map);
        }

        public static IResizeableFilter operator +(IFilter filter, RenderChain map)
        {
            return filter.Apply(map).MakeResizeable();
        }

        public static RenderChain operator +(RenderChain f, RenderChain g)
        {
            return (RenderChain)(filter => (filter + f) + g);
        }

        #endregion

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
                return Path.Combine(PathHelper.GetDirectoryName(asmPath), "Extensions", "RenderScripts", ShaderPath);
            }
        }

        protected IShader CompileShader(string shaderFileName)
        {
            return ShaderCache<IShader>.Add(Path.Combine(ShaderDataFilePath, shaderFileName),
                s => Renderer.CompileShader(s));
        }

        protected IShader LoadShader(string shaderFileName)
        {
            return ShaderCache<IShader>.Add(Path.Combine(ShaderDataFilePath, shaderFileName),
                Renderer.LoadShader);
        }

        protected IShader11 CompileShader11(string shaderFileName, string profile)
        {
            return CompileShader11(shaderFileName, "main", profile);
        }

        protected IShader11 CompileShader11(string shaderFileName, string entryPoint, string profile)
        {
            return ShaderCache<IShader11>.Add(Path.Combine(ShaderDataFilePath, shaderFileName),
                s => Renderer.CompileShader11(s, entryPoint, profile));
        }

        protected IKernel CompileClKernel(string sourceFileName, string entryPoint, string options = null)
        {
            return ShaderCache<IKernel>.Add(Path.Combine(ShaderDataFilePath, sourceFileName),
                s => Renderer.CompileClKernel(s, entryPoint, options));
        }

        protected IShader11 LoadShader11(string shaderFileName)
        {
            return ShaderCache<IShader11>.Add(Path.Combine(ShaderDataFilePath, shaderFileName),
                Renderer.LoadShader11);
        }

        #endregion

        #region Size Calculations

        protected bool IsDownscalingFrom(TextureSize size)
        {
            return !IsNotScalingFrom(size) && !IsUpscalingFrom(size);
        }

        protected bool IsNotScalingFrom(TextureSize size)
        {
            return size == Renderer.TargetSize;
        }

        protected bool IsUpscalingFrom(TextureSize size)
        {
            var targetSize = Renderer.TargetSize;
            return targetSize.Width > size.Width || targetSize.Height > size.Height;
        }

        protected bool IsDownscalingFrom(IFilter chain)
        {
            return IsDownscalingFrom(chain.OutputSize);
        }

        protected bool IsNotScalingFrom(IFilter chain)
        {
            return IsNotScalingFrom(chain.OutputSize);
        }

        protected bool IsUpscalingFrom(IFilter chain)
        {
            return IsUpscalingFrom(chain.OutputSize);
        }

        #endregion

        #region Resource Disposal Methods

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
        }

        public virtual void RenderScriptDisposed()
        {
        }

        #endregion
    }

    public class StaticChain : RenderChain
    {
        private readonly Func<IResizeableFilter, IFilter> m_Compiler;

        public StaticChain(Func<IResizeableFilter, IFilter> compiler)
        {
            m_Compiler = compiler;
        }

        public override IFilter CreateFilter(IResizeableFilter sourceFilter)
        {
            return m_Compiler(sourceFilter);
        }
    }
}