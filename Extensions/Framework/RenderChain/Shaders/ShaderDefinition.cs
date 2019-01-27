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
using Mpdn.OpenCl;
using Mpdn.RenderScript;

namespace Mpdn.Extensions.Framework.RenderChain.Shaders
{
    public interface IShaderDefinition<TShader>
        where TShader : IShaderBase
    {
        void Compile(out TShader shader);
    }

    public static class DefinitionHelper
    {
        public static TShader Compile<TShader>(this IShaderDefinition<TShader> definition)
            where TShader : IShaderBase
        {
            TShader shader;
            definition.Compile(out shader);
            return shader;
        }

        public static ShaderFromFile FromFile(string shaderFileName, string profile = "ps_3_0", string entryPoint = "main", string compilerOptions = null)
        {
            return new ShaderFromFile(shaderFileName, profile, entryPoint, compilerOptions);
        }

        public static ShaderFromString FromString(string shadercode, string profile = "ps_3_0", string entryPoint = "main", string compilerOptions = null)
        {
            return new ShaderFromString(shadercode, profile, entryPoint, compilerOptions);
        }

        public static ShaderFromByteCode FromByteCode(string bytecodeFileName)
        {
            return new ShaderFromByteCode(bytecodeFileName);
        }
    }

    public class CachedDefinition<TShader> : IShaderDefinition<TShader>
        where TShader : class, IShaderBase
    {
        private readonly IShaderDefinition<TShader> m_Definition;

        private TShader m_CachedShader;

        public CachedDefinition(IShaderDefinition<TShader> definition)
        {
            m_Definition = definition;
        }

        public void Compile(out TShader shader)
        {
            shader = m_CachedShader ?? (m_CachedShader = m_Definition.Compile());
        }
    }

    public class ShaderFromFile : IShaderDefinition<IShader>, IShaderDefinition<IShader11>, IShaderDefinition<IKernel>
    { 
        protected readonly string ShaderFilename;
        protected readonly string Profile;
        protected readonly string EntryPoint;
        protected readonly string CompilerOptions;

        public ShaderFromFile(string shaderFilename, string profile = "ps_3_0", string entryPoint = "main", string compilerOptions = null)
        {
            ShaderFilename = shaderFilename;
            Profile = profile;
            EntryPoint = entryPoint;
            CompilerOptions = compilerOptions;
        }

        public void Compile(out IShader shader)
        {
            shader = ShaderCache.CompileShader(ShaderFilename, Profile, EntryPoint, CompilerOptions);
        }

        public void Compile(out IShader11 shader)
        {
            shader = ShaderCache.CompileShader11(ShaderFilename, Profile, EntryPoint, CompilerOptions);
        }

        public void Compile(out IKernel kernel)
        {
            kernel = ShaderCache.CompileClKernel(ShaderFilename, EntryPoint, CompilerOptions);
        }
    }

    public class ShaderFromByteCode : IShaderDefinition<IShader>, IShaderDefinition<IShader11>
    {
        protected readonly string BytecodeFilename;

        public ShaderFromByteCode(string bytecodeFilename)
        {
            BytecodeFilename = bytecodeFilename;
        }

        public void Compile(out IShader shader)
        {
            shader = ShaderCache.LoadShader(BytecodeFilename);
        }

        public void Compile(out IShader11 shader)
        {
            shader = ShaderCache.LoadShader11(BytecodeFilename);
        }
    }

    public class ShaderFromString : IShaderDefinition<IShader>, IShaderDefinition<IShader11>, IShaderDefinition<IKernel>
    {
        protected readonly string Shadercode;
        protected readonly string Profile;
        protected readonly string EntryPoint;
        protected readonly string CompilerOptions;

        public ShaderFromString(string shadercode, string profile = "ps_3_0", string entryPoint = "main", string compilerOptions = null)
        {
            Shadercode = shadercode;
            Profile = profile;
            EntryPoint = entryPoint;
            CompilerOptions = compilerOptions;
        }

        public void Compile(out IShader shader)
        {
            shader = Renderer.CompileShaderFromString(Shadercode, EntryPoint, Profile, CompilerOptions);
        }

        public void Compile(out IShader11 shader)
        {
            shader = Renderer.CompileShader11FromString(Shadercode, EntryPoint, Profile, CompilerOptions);
        }

        public void Compile(out IKernel kernel)
        {
            kernel = Renderer.CompileClKernelFromString(Shadercode, Profile, CompilerOptions);
        }
    }
}