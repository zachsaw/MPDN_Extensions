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
using Mpdn.Extensions.Framework.Chain;
using Mpdn.OpenCl;
using Mpdn.RenderScript;

namespace Mpdn.Extensions.Framework.RenderChain
{
    public abstract class RenderChain : Chain<IFilter>
    {
        protected RenderChain()
        {
            ShaderCache.Load();
        }

        protected abstract IFilter CreateFilter(IFilter input);

        public sealed override IFilter Process(IFilter input)
        {
            IFilter output = CreateFilter(input);

            if (output == input)
                return input;

            if (output.Tag.IsEmpty())
            {
                output.AddTag(Status);
                output.Tag.AddInput(input);
            }

            return output
                .AddTaggedResizer()
                .Tagged(new TemporaryTag("Resizer"));
        }

        #region Status

        public virtual string Status
        {
            get { return GetType().Name; }
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
    }
}