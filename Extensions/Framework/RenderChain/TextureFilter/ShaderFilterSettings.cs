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
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Mpdn.RenderScript;
using SharpDX;

namespace Mpdn.Extensions.Framework.RenderChain.TextureFilter
{
    public struct ShaderFilterSettings<T> : IShaderFilterSettings<T>
        where T : IShaderBase
    {
        public T Shader { get; set; }
        public bool LinearSampling { get; set; }
        public bool[] PerTextureLinearSampling { get; set; }
        public Func<TextureSize, TextureSize> Transform { get; set; }
        public TextureFormat Format { get; set; }
        public int SizeIndex { get; set; }
        public ArgumentList Arguments { get; set; }
        public ArgumentList.Entry this[string identifier]
        {
            get { return Arguments[identifier]; }
            set { Arguments[identifier] = value; }
        }

        public ShaderFilterSettings(T shader)
        {
            Shader = shader;
            LinearSampling = false;
            PerTextureLinearSampling = new bool[0];
            Transform = (s => s);
            Format = Renderer.RenderQuality.GetTextureFormat();
            SizeIndex = 0;
            Arguments = new ArgumentList();
        }

        public static implicit operator ShaderFilterSettings<T>(T shader)
        {
            return new ShaderFilterSettings<T>(shader);
        }

        public IShaderFilterSettings<T> Configure(bool? linearSampling = null,
            ArgumentList arguments = null, Func<TextureSize, TextureSize> transform = null, int? sizeIndex = null, 
            TextureFormat? format = null, IEnumerable<bool> perTextureLinearSampling = null)
        {
            return new ShaderFilterSettings<T>(Shader)
            {
                Transform = transform ?? Transform,
                LinearSampling = linearSampling ?? LinearSampling,
                Format = format ?? Format,
                SizeIndex = sizeIndex ?? SizeIndex,
                Arguments = arguments ?? Arguments,
                PerTextureLinearSampling = perTextureLinearSampling != null ? perTextureLinearSampling.ToArray() : PerTextureLinearSampling
            };
        }   
    }
}