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

namespace Mpdn.Extensions.Framework.RenderChain.Shader
{
    public interface IShaderConfig
    {
        IShaderHandle GetHandle();
    }

    public abstract class GenericShaderConfig<TShader> : IShaderConfig, IShaderParameters
        where TShader : IShaderBase
    {
        public IShaderDefinition<TShader> Definition { get; set; }

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

        public abstract IShaderHandle GetHandle();

        protected GenericShaderConfig(IShaderDefinition<TShader> definition)
        {
            Definition = definition;
            LinearSampling = false;
            PerTextureLinearSampling = new bool[0];
            Transform = (s => s);
            Format = Renderer.RenderQuality.GetTextureFormat();
            SizeIndex = 0;
            Arguments = new ArgumentList();
        }

        protected GenericShaderConfig(GenericShaderConfig<TShader> config)
        {
            Definition = config.Definition;
            Transform = config.Transform;
            LinearSampling = config.LinearSampling;
            PerTextureLinearSampling = config.PerTextureLinearSampling;
            Transform = config.Transform;
            Format = config.Format;
            SizeIndex = config.SizeIndex;
            Arguments = config.Arguments;
        }
    }

    public class Shader : GenericShaderConfig<IShader>
    {
        public Shader(IShaderDefinition<IShader> definition) : base(definition) { }

        public Shader(Shader config) : base(config) { }

        public override IShaderHandle GetHandle()
        {
            return new ShaderHandle(this, Definition);
        }
    }

    public class Shader11 : GenericShaderConfig<IShader11>
    {
        public Shader11(IShaderDefinition<IShader11> definition) : base(definition) { }

        public Shader11(Shader11 config) : base(config) { }

        public override IShaderHandle GetHandle()
        {
            return new Shader11Handle(this, Definition);
        }
    }

    public class ComputeShader : Shader11, IComputeShaderParameters
    {
        public int ThreadGroupX { get; set; }
        public int ThreadGroupY { get; set; }
        public int ThreadGroupZ { get; set; }

        public ComputeShader(IShaderDefinition<IShader11> definition, int threadGroupX, int threadGroupY, int threadGroupZ) : base(definition)
        {
            ThreadGroupX = threadGroupX;
            ThreadGroupY = threadGroupY;
            ThreadGroupZ = threadGroupZ;
        }

        public ComputeShader(ComputeShader config) : base(config)
        {
            ThreadGroupX = config.ThreadGroupX;
            ThreadGroupY = config.ThreadGroupY;
            ThreadGroupZ = config.ThreadGroupZ;
        }

        public override IShaderHandle GetHandle()
        {
            return new ComputeShaderHandle(this, Definition);
        }
    }

    public class ClKernel : GenericShaderConfig<IKernel>, IClKernelParameters
    {
        public int[] GlobalWorkSizes { get; set; }
        public int[] LocalWorkSizes { get; set; }

        public ClKernel(IShaderDefinition<IKernel> definition, int[] globalWorkSizes, int[] localWorkSizes) : base(definition)
        {
            GlobalWorkSizes = globalWorkSizes;
            LocalWorkSizes = localWorkSizes;
        }

        public ClKernel(ClKernel config) : base(config)
        {
            GlobalWorkSizes = config.GlobalWorkSizes;
            LocalWorkSizes = config.LocalWorkSizes;
        }

        public override IShaderHandle GetHandle()
        {
            return new ClKernelHandle(this, Definition);
        }
    }
}
