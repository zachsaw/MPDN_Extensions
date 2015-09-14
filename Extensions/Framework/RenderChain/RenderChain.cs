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
using Mpdn.RenderScript;

namespace Mpdn.Extensions.Framework.RenderChain
{
    public abstract class RenderChain
    {
        private Func<string> m_Status;

        protected RenderChain()
        {
            ShaderCache.Load();
            m_Status = Inactive;
        }

        protected abstract IFilter CreateFilter(IFilter input);

        public IFilter MakeFilter(IFilter filter)
        {
            Status = null;
            IFilter result;
            try
            {
                result = CreateFilter(filter);
            }
            catch (Exception)
            {
                Status = Inactive;
                throw;
            }

            var activeStatus = Status ?? Active;
            Status = () => result.Active && (!filter.Active || filter.Compile() != result.Compile())
                ? activeStatus().AppendStatus(result.Compile().ResizerDescription())
                : Inactive();

            return result;
        }

        #region Status

        public virtual Func<string> Status
        {
            get { return m_Status; }
            set { m_Status = value; }
        }

        public virtual string Active()
        {
            return GetType().Name;
        }

        public string Inactive()
        {
            return string.Empty;
        }

        #endregion

        #region Operators

        public static readonly RenderChain Identity = RenderChainUi.Identity.Chain;

        public static implicit operator Func<IFilter, IFilter>(RenderChain map)
        {
            return map.MakeFilter;
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

        protected IShader CompileShader(string shaderFileName, string profile = "ps_3_0", string entryPoint = "main", string macroDefinitions = null)
        {
            return ShaderCache.CompileShader(Path.Combine(ShaderDataFilePath, shaderFileName), profile, entryPoint, macroDefinitions);
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

        #region Implicit Shader Conversion

        public static implicit operator RenderChain(ShaderFilterSettings<IShader> shaderSettings)
        {
            return (RenderChain)(f => new ShaderFilter(shaderSettings, f));
        }

        public static implicit operator RenderChain(ShaderFilterSettings<IShader11> shaderSettings)
        {
            return (RenderChain)(f => new Shader11Filter(shaderSettings, f));
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

        #region Resource Management Methods

        /// <summary>
        /// Called when user activates a render script
        /// </summary>
        public virtual void Initialize()
        {
        }

        /// <summary>
        /// Dispose any unmanaged resource that shouldn't be retained when user selects a new render script
        /// </summary>
        public virtual void Reset()
        {
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

        protected override IFilter CreateFilter(IFilter input)
        {
            return m_Compiler(input);
        }
    }
}