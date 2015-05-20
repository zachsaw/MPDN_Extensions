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
using System.Diagnostics;
using Mpdn.OpenCl;
using TransformFunc = System.Func<System.Drawing.Size, System.Drawing.Size>;

namespace Mpdn.RenderScript
{
    public abstract class RenderChain : IDisposable
    {
        public abstract IFilter CreateFilter(IFilter input);

        #region Operators

        public static RenderChain Identity = new StaticChain(x => x);

        public static implicit operator Func<IFilter, IFilter>(RenderChain map)
        {
            return map.CreateFilter;
        }

        public static implicit operator RenderChain(Func<IFilter, IFilter> map)
        {
            return new StaticChain(map);
        }

        public static IFilter operator +(IFilter filter, RenderChain map)
        {
            return filter.Apply(map);
        }

        public static RenderChain operator +(RenderChain f, RenderChain g)
        {
            return (RenderChain) (filter => (filter + f) + g);
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

        protected IShader CompileShader(string shaderFileName, string entryPoint = "main", string macroDefinitions = null)
        {
            return ShaderCache.CompileShader(Path.Combine(ShaderDataFilePath, shaderFileName), entryPoint, macroDefinitions);
        }

        protected IShader11 CompileShader11(string shaderFileName, string profile, string entryPoint = "main", string macroDefinitions = null)
        {
            return ShaderCache.CompileShader11(Path.Combine(ShaderDataFilePath, shaderFileName), profile, entryPoint, macroDefinitions);
        }

        protected IKernel CompileClKernel(string sourceFileName, string entryPoint, string options = null)
        {
            return ShaderCache.CompileClKernel(Path.Combine(ShaderDataFilePath, sourceFileName), entryPoint, options);
        }

        protected IShader LoadShader(string shaderFileName)
        {
            return ShaderCache.LoadShader(Path.Combine(ShaderDataFilePath, shaderFileName));
        }

        protected IShader11 LoadShader11(string shaderFileName)
        {
            return ShaderCache.LoadShader11(Path.Combine(ShaderDataFilePath, shaderFileName));
        }

        #endregion

        #region Size Calculations

        public bool IsDownscalingFrom(TextureSize size)
        {
            return !IsNotScalingFrom(size) && !IsUpscalingFrom(size);
        }

        public bool IsNotScalingFrom(TextureSize size)
        {
            return size == Renderer.TargetSize;
        }

        public bool IsUpscalingFrom(TextureSize size)
        {
            var targetSize = Renderer.TargetSize;
            return targetSize.Width > size.Width || targetSize.Height > size.Height;
        }

        public bool IsDownscalingFrom(IFilter chain)
        {
            return IsDownscalingFrom(chain.OutputSize);
        }

        public bool IsNotScalingFrom(IFilter chain)
        {
            return IsNotScalingFrom(chain.OutputSize);
        }

        public bool IsUpscalingFrom(IFilter chain)
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

        ~RenderChain()
        {
            Dispose(false);
        }

        #endregion

        #region Error Handling

        public IFilter CreateSafeFilter(IFilter input)
        {
            try
            {
                return CreateFilter(input);
            }
            catch (Exception ex)
            {
                return DisplayError(ex);
            }
        }

        private IFilter DisplayError(Exception e)
        {
            var message = ErrorMessage(e);
            Trace.WriteLine(message);
            return new TextFilter(message);
        }

        protected static Exception InnerMostException(Exception e)
        {
            while (e.InnerException != null)
            {
                e = e.InnerException;
            }

            return e;
        }

        private string ErrorMessage(Exception e)
        {
            var ex = InnerMostException(e);
            return string.Format("Error in {0}:\r\n\r\n{1}\r\n\r\n~\r\nStack Trace:\r\n{2}",
                    GetType().Name, ex.Message, ex.StackTrace);
        }

        #endregion
    }

    public class StaticChain : RenderChain
    {
        private readonly Func<IFilter, IFilter> m_Compiler;

        public StaticChain(Func<IFilter, IFilter> compiler)
        {
            m_Compiler = compiler;
        }

        public override IFilter CreateFilter(IFilter input)
        {
            return m_Compiler(input);
        }
    }
}