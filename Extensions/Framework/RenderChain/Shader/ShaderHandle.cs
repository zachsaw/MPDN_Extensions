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
using Mpdn.OpenCl;
using Mpdn.RenderScript;
using SharpDX;

namespace Mpdn.Extensions.Framework.RenderChain.Shader
{
    public interface IShaderHandle
    {
        TextureFormat OutputFormat { get; }
        TextureSize CalcSize(IList<TextureSize> inputSizes);

        void Initialize();
        void LoadArguments(IList<IBaseTexture> inputs, ITargetTexture output);
        void Render();
    }

    public abstract class BaseShaderHandle<TShader> : IShaderHandle
        where TShader : IShaderBase
    {
        private readonly IShaderDefinition<TShader> m_Definition;
        protected TShader Shader;

        public abstract void Render();
        public abstract void LoadArguments(IList<IBaseTexture> inputs, ITargetTexture output);

        protected BaseShaderHandle(IShaderParameters parameters, IShaderDefinition<TShader> definition)
        {
            Transform = parameters.Transform;
            Format = parameters.Format;
            SizeIndex = parameters.SizeIndex;
            Arguments = new ArgumentList(parameters.Arguments);
            LinearSampling = parameters.LinearSampling;
            PerTextureLinearSampling = parameters.PerTextureLinearSampling;

            m_Definition = definition;
        }

        protected Func<TextureSize, TextureSize> Transform;
        protected TextureFormat Format;
        protected int SizeIndex;

        protected bool LinearSampling { get; set; }
        protected bool[] PerTextureLinearSampling { get; set; }
        protected ArgumentList Arguments { get; set; }

        public TextureFormat OutputFormat { get { return Format; } }

        public TextureSize CalcSize(IList<TextureSize> inputSizes)
        {
            if (SizeIndex < 0 || SizeIndex >= inputSizes.Count)
            {
                throw new IndexOutOfRangeException(string.Format("No valid input filter at index {0}", SizeIndex));
            }

            return Transform(inputSizes[SizeIndex]);
        }

        public void Initialize()
        {
            Shader = m_Definition.Compile();
        }
    }

    public abstract class GenericShaderHandle<TShader> : BaseShaderHandle<TShader>
        where TShader : IShaderBase
    {
        protected abstract void SetTextureConstant(int index, IBaseTexture texture, bool linearSampling);
        protected abstract void SetConstant(string identifier, Vector4 value);

        protected GenericShaderHandle(IShaderParameters parameters, IShaderDefinition<TShader> definition)
            : base(parameters, definition)
        { }

        protected ITargetTexture Target;

        protected virtual void LoadCustomConstants() { }

        public override void LoadArguments(IList<IBaseTexture> inputs, ITargetTexture output)
        {
            LoadTextureConstants(inputs, output);
            LoadShaderConstants();
            LoadCustomConstants();
        }

        protected virtual void LoadTextureConstants(IList<IBaseTexture> inputs, ITargetTexture output)
        {
            var index = 0;
            foreach (var input in inputs)
            {
                var linearSampling = PerTextureLinearSampling
                                         .Cast<bool?>()
                                         .ElementAtOrDefault(index)
                                     ?? LinearSampling;
                SetTextureConstant(index, input, linearSampling);
                LoadSizeConstant(index.ToString(), input.GetSize());
                index++;
            }

            /* Add Output Size */
            LoadSizeConstant("Output", output.GetSize());
            SetOutputTexture(output);
        }

        protected void LoadSizeConstant(string identifier, TextureSize size)
        {
            if (size.Is3D)
                SetConstant(string.Format("size3d{0}", identifier), new Vector4(size.Width, size.Height, size.Depth, 0));
            else
                SetConstant(string.Format("size{0}", identifier), new Vector4(size.Width, size.Height, 1.0f / size.Width, 1.0f / size.Height));
        }

        protected void LoadShaderConstants()
        {
            foreach (var argument in Arguments)
                SetConstant(argument.Key, argument.Value);
        }

        protected virtual void SetOutputTexture(ITargetTexture targetTexture)
        {
            Target = targetTexture;
        }
    }

    public class ShaderHandle : GenericShaderHandle<IShader>
    {
        public ShaderHandle(IShaderParameters parameters, IShaderDefinition<IShader> definition)
            : base(parameters, definition)
        { }

        protected override void SetTextureConstant(int index, IBaseTexture texture, bool linearSampling)
        {
            if (texture is ITexture2D)
            {
                var tex = (ITexture2D)texture;
                Shader.SetTextureConstant(index, tex, linearSampling, false);
            }
            else
            {
                var tex = (ITexture3D)texture;
                Shader.SetTextureConstant(index, tex, linearSampling, false);
            }
        }

        protected override void SetConstant(string identifier, Vector4 value)
        {
            Shader.SetConstant(identifier, value, false);
        }

        public override void Render()
        {
            Renderer.Render(Target, Shader);
        }
    }

    public class Shader11Handle : GenericShaderHandle<IShader11>
    {
        private readonly IList<IDisposable> m_Buffers = new List<IDisposable>();

        public Shader11Handle(IShaderParameters parameters, IShaderDefinition<IShader11> definition)
            : base(parameters, definition)
        { }

        protected override void SetTextureConstant(int index, IBaseTexture texture, bool linearSampling)
        {
            if (texture is ITexture2D)
            {
                var tex = (ITexture2D)texture;
                Shader.SetTextureConstant(index, tex, linearSampling, false);
            }
            else
            {
                var tex = (ITexture3D)texture;
                Shader.SetTextureConstant(index, tex, linearSampling, false);
            }
        }

        protected override void SetConstant(string identifier, Vector4 value)
        {
            var buffer = Renderer.CreateConstantBuffer(value);
            m_Buffers.Add(buffer);
            Shader.SetConstantBuffer(identifier, buffer, false);
        }

        protected override void SetOutputTexture(ITargetTexture targetTexture)
        {
            Target = targetTexture;
        }

        public override void Render()
        {
            Renderer.Render(Target, Shader);

            DisposeHelper.DisposeElements(m_Buffers);
            m_Buffers.Clear();
        }
    }

    public class ComputeShaderHandle : Shader11Handle
    {
        protected int ThreadGroupX { get; set; }
        protected int ThreadGroupY { get; set; }
        protected int ThreadGroupZ { get; set; }

        public ComputeShaderHandle(IComputeShaderParameters parameters, IShaderDefinition<IShader11> definition)
            : base(parameters, definition)
        {
            ThreadGroupX = parameters.ThreadGroupX;
            ThreadGroupY = parameters.ThreadGroupY;
            ThreadGroupZ = parameters.ThreadGroupZ;
        }

        public override void Render()
        {
            Renderer.Compute(Target, Shader, ThreadGroupX, ThreadGroupY, ThreadGroupZ);
        }
    }

    public class ClKernelHandle : BaseShaderHandle<IKernel>
    {
        public int[] GlobalWorkSizes { get; set; }
        public int[] LocalWorkSizes { get; set; }

        protected virtual void LoadAdditionalInputs(int currentIndex) { }

        public ClKernelHandle(IClKernelParameters parameters, IShaderDefinition<IKernel> definition)
            : base(parameters, definition)
        {
            GlobalWorkSizes = parameters.GlobalWorkSizes;
            LocalWorkSizes = parameters.LocalWorkSizes;
        }

        public override void LoadArguments(IList<IBaseTexture> inputs, ITargetTexture output)
        {
            IList<ITexture2D> inputs2D;
            try
            {
                inputs2D = inputs.Cast<ITexture2D>().ToList();
            }
            catch (InvalidCastException e)
            {
                throw new ArgumentException("OpenCL only supports 2D textures.", e);
            }

            var index = 0;
            foreach (var input in inputs2D)
                Shader.SetInputTextureArg(index++, input);

            Shader.SetOutputTextureArg(index++, output);
            LoadAdditionalInputs(index);
        }

        public override void Render()
        {
            if (GlobalWorkSizes == null)
                throw new ArgumentNullException("OpenCL global workgroup sizes not set.");

            Renderer.RunClKernel(Shader, GlobalWorkSizes, LocalWorkSizes);
        }
    }
}
