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

using System;
using System.Collections.Generic;
using System.Linq;
using Mpdn.Extensions.Framework.Filter;
using Mpdn.Extensions.Framework.RenderChain.TextureFilter;

namespace Mpdn.Extensions.Framework.RenderChain.Shader
{
    public class ShaderFilter : TextureFilter.TextureFilter
    {
        protected readonly IShaderHandle Shader;

        public ShaderFilter(IShaderConfig config, params IFilter<ITextureOutput<IBaseTexture>>[] inputFilters)
            : this(config.GetHandle(), inputFilters)
        { }

        public ShaderFilter(IShaderHandle shader, params IFilter<ITextureOutput<IBaseTexture>>[] inputFilters)
            : base(shader.CalcSize(inputFilters.Select(f => f.Size()).ToList()), shader.OutputFormat, inputFilters)
        {
            Shader = shader;
        }

        protected override void Initialize()
        {
            base.Initialize();
            Shader.Initialize();
        }

        protected override void Render(IList<ITextureOutput<IBaseTexture>> inputs)
        {
            Shader.LoadArguments(inputs.Select(x => x.Texture).ToList(), Target.Texture);
            Shader.Render();
        }
    }
}