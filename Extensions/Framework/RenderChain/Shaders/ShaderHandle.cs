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
using Mpdn.Extensions.Framework.Filter;
using Mpdn.Extensions.Framework.RenderChain.Filters;
using Shiandow.Lending;

namespace Mpdn.Extensions.Framework.RenderChain.Shaders
{
    public interface IShaderHandle : IMultiProcess<ITextureFilter<IBaseTexture>, ITextureFilter>, IDisposable { }

    public abstract class BaseShaderHandle<TShader> : IMultiProcess<ITextureFilter<IBaseTexture>, ITextureFilter>, IShaderHandle
        where TShader : IShaderBase
    {
        protected abstract void RenderOutput(ITargetTexture output);
        protected abstract void LoadArguments(IList<IBaseTexture> inputs, ITargetTexture output);

        #region Implementation

        protected readonly TShader Shader;

        protected readonly Func<TextureSize, TextureSize> Transform;
        protected readonly TextureFormat Format;
        protected readonly int SizeIndex;

        protected readonly bool LinearSampling;
        protected readonly bool[] PerTextureLinearSampling;
        protected readonly ArgumentList Arguments;

        private readonly IShaderDefinition<TShader> m_Definition;

        protected BaseShaderHandle(IShaderParameters parameters, IShaderDefinition<TShader> definition)
        {
            Transform = parameters.Transform;
            Format = parameters.Format;
            SizeIndex = parameters.SizeIndex;
            Arguments = new ArgumentList(parameters.Arguments);
            LinearSampling = parameters.LinearSampling;
            PerTextureLinearSampling = parameters.PerTextureLinearSampling;

            m_Definition = definition;
            Shader = m_Definition.Compile();
        }

        private ITextureOutput Allocate(IEnumerable<ITextureDescription> inputs)
        {
            return new TextureOutput(CalcSize(inputs.Select(x => x.Size)), Format);
        }

        private TextureSize CalcSize(IEnumerable<TextureSize> inputSizes)
        {
            var sizes = inputSizes.ToList();
            if (SizeIndex < 0 || SizeIndex >= sizes.Count)
            {
                throw new IndexOutOfRangeException(string.Format("No valid input filter at index {0}", SizeIndex));
            }

            return Transform(sizes[SizeIndex]);
        }

        #endregion

        #region IProcess Implementation

        public ITextureFilter ApplyTo(IEnumerable<ITextureFilter<IBaseTexture>> inputs)
        {
            var folded = inputs.Fold();
            return new TextureFilter(
                from _ in FilterBaseHelper.Return(this)
                from values in folded
                select Allocate(folded.Output).Do(Render, values));
        }

        public void Render(IEnumerable<IBaseTexture> inputs, ITargetTexture output)
        {
            LoadArguments(inputs.ToList(), output);
            RenderOutput(output);
        }

        #endregion

        #region Resource Management

        ~BaseShaderHandle()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing) { }

        #endregion
    }

    public abstract class GenericShaderHandle<TShader> : BaseShaderHandle<TShader>
        where TShader : IShaderBase
    {
        protected abstract void SetTextureConstant(int index, IBaseTexture texture, bool linearSampling);
        protected abstract void SetConstant(string identifier, Vector4 value);

        #region Implementation

        protected GenericShaderHandle(IShaderParameters parameters, IShaderDefinition<TShader> definition)
            : base(parameters, definition)
        { }

        protected override void LoadArguments(IList<IBaseTexture> inputs, ITargetTexture output)
        {
            LoadTextureConstants(inputs, output);
            LoadShaderConstants();
        }

        protected void LoadTextureConstants(IList<IBaseTexture> inputs, ITargetTexture output)
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

        #endregion
    }

    public class ShaderHandle : GenericShaderHandle<IShader>
    {
        public ShaderHandle(IShaderParameters parameters, IShaderDefinition<IShader> definition)
            : base(parameters, definition)
        { }

        #region Implementation

        protected sealed override void SetTextureConstant(int index, IBaseTexture texture, bool linearSampling)
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

        protected sealed override void SetConstant(string identifier, Vector4 value)
        {
            Shader.SetConstant(identifier, value, false);
        }

        protected sealed override void RenderOutput(ITargetTexture output)
        {
            Renderer.Render(output, Shader);
        }

        #endregion
    }

    public abstract class BaseShader11Handle : GenericShaderHandle<IShader11>
    {
        protected abstract void Render(ITargetTexture output);

        #region Implementation

        private readonly IList<IDisposable> m_Buffers = new List<IDisposable>();

        public BaseShader11Handle(IShaderParameters parameters, IShaderDefinition<IShader11> definition)
            : base(parameters, definition)
        { }

        protected sealed override void SetTextureConstant(int index, IBaseTexture texture, bool linearSampling)
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

        protected sealed override void SetConstant(string identifier, Vector4 value)
        {
            var buffer = Renderer.CreateConstantBuffer(value);
            m_Buffers.Add(buffer);
            Shader.SetConstantBuffer(identifier, buffer, false);
        }

        protected sealed override void RenderOutput(ITargetTexture output)
        {
            Render(output);

            DisposeHelper.DisposeElements(m_Buffers);
            m_Buffers.Clear();
        }

        #endregion
    }

    public class Shader11Handle : BaseShader11Handle
    {
        public Shader11Handle(IShaderParameters parameters, IShaderDefinition<IShader11> definition)
            : base(parameters, definition)
        { }

        protected sealed override void Render(ITargetTexture output)
        {
            Renderer.Render(output, Shader);
        }
    }

    public class ComputeShaderHandle : BaseShader11Handle
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

        protected sealed override void Render(ITargetTexture output)
        {
            Renderer.Compute(output, Shader, ThreadGroupX, ThreadGroupY, ThreadGroupZ);
        }
    }

    public class ClKernelHandle : BaseShaderHandle<IKernel>
    {
        protected int[] GlobalWorkSizes { get; set; }
        protected int[] LocalWorkSizes { get; set; }

        protected virtual void LoadAdditionalInputs(int currentIndex) { }

        public ClKernelHandle(IClKernelParameters parameters, IShaderDefinition<IKernel> definition)
            : base(parameters, definition)
        {
            GlobalWorkSizes = parameters.GlobalWorkSizes;
            LocalWorkSizes = parameters.LocalWorkSizes;
        }

        #region Implementation

        protected override void LoadArguments(IList<IBaseTexture> inputs, ITargetTexture output)
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

        protected sealed override void RenderOutput(ITargetTexture output)
        {
            if (GlobalWorkSizes == null)
                throw new ArgumentNullException("OpenCL global workgroup sizes not set.");

            Renderer.RunClKernel(Shader, GlobalWorkSizes, LocalWorkSizes);
        }

        #endregion
    }
}
