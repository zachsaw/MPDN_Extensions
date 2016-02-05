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

    public class ArgumentList : IEnumerable<KeyValuePair<string, ArgumentList.Entry>>
    {
        private readonly IDictionary<string, Entry> m_Arguments;

        public ArgumentList()
            : this(new Dictionary<string, Entry>())
        { }

        public ArgumentList(IDictionary<string, Entry> arguments)
        {
            m_Arguments = arguments;
        }

        #region Implementation 

        // Allow shader arguments to be accessed individually
        public Entry this[string identifier]
        {
            get { return m_Arguments[identifier]; }
            set { m_Arguments[identifier] = value; }
        }

        public ArgumentList Merge(ArgumentList other)
        {
            var dict = new Dictionary<String, Entry>(m_Arguments);
            foreach (var pair in other)
                dict[pair.Key] = pair.Value;

            return new ArgumentList(dict);
        }

        public IEnumerator<KeyValuePair<string, Entry>> GetEnumerator()
        {
            return m_Arguments.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        #endregion

        #region Operators

        public static implicit operator ArgumentList(Dictionary<string, Entry> arguments)
        {
            return new ArgumentList(arguments);
        }

        public static implicit operator ArgumentList(Dictionary<string, Vector4> arguments)
        {
            return arguments.ToDictionary(x => x.Key, x => (Entry)x.Value);
        }

        public static implicit operator ArgumentList(Dictionary<string, Vector3> arguments)
        {
            return arguments.ToDictionary(x => x.Key, x => (Entry)x.Value);
        }

        public static implicit operator ArgumentList(Dictionary<string, Vector2> arguments)
        {
            return arguments.ToDictionary(x => x.Key, x => (Entry)x.Value);
        }

        public static implicit operator ArgumentList(Dictionary<string, float> arguments)
        {
            return arguments.ToDictionary(x => x.Key, x => (Entry)x.Value);
        }

        public static implicit operator ArgumentList(float[] arguments)
        {
            var dict = new Dictionary<string, Vector4>();
            for (var i = 0; 4 * i < arguments.Length; i++)
            {
                dict.Add(string.Format("args{0}", i),
                    new Vector4(arguments.ElementAtOrDefault(4 * i), arguments.ElementAtOrDefault(4 * i + 1), arguments.ElementAtOrDefault(4 * i + 2), arguments.ElementAtOrDefault(4 * i + 3)));
            }
            return dict;
        }

        #endregion

        #region Auxilary Types

        public struct Entry
        {
            private readonly Vector4 m_Value;

            private Entry(Vector4 value)
            {
                m_Value = value;
            }

            #region Operators

            public static implicit operator Vector4(Entry argument)
            {
                return argument.m_Value;
            }

            public static implicit operator Entry(Vector4 argument)
            {
                return new Entry(argument);
            }

            public static implicit operator Entry(Vector3 argument)
            {
                return new Vector4(argument, 0.0f);
            }

            public static implicit operator Entry(Vector2 argument)
            {
                return new Vector4(argument, 0.0f, 0.0f);
            }

            public static implicit operator Entry(float argument)
            {
                return new Vector4(argument);
            }

            #endregion
        }

        #endregion
    }

}