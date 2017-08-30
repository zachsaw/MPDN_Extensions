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
using Mpdn.OpenCl;
using Mpdn.RenderScript;
using SharpDX;

namespace Mpdn.Extensions.Framework.RenderChain.TextureFilter
{
    using TransformFunc = Func<TextureSize, TextureSize>;
    using IBaseTextureFilter = IFilter<ITextureOutput<IBaseTexture>>;

    public abstract class GenericShaderFilter<T> : TextureFilter 
        where T : IShaderBase
    {
        protected GenericShaderFilter(T shader, params IBaseTextureFilter[] inputFilters)
            : this((ShaderFilterSettings<T>) shader, inputFilters)
        { }

        protected static TextureSize CalcSize(IShaderFilterSettings<T> settings, params IBaseTextureFilter[] inputFilters)
        {
            if (settings.SizeIndex < 0 || settings.SizeIndex >= inputFilters.Length || inputFilters[settings.SizeIndex] == null)
            {
                throw new IndexOutOfRangeException(string.Format("No valid input filter at index {0}", settings.SizeIndex));
            }

            return settings.Transform(inputFilters[settings.SizeIndex].Size());
        }

        protected GenericShaderFilter(IShaderFilterSettings<T> settings, params IBaseTextureFilter[] inputFilters)
            : base(CalcSize(settings, inputFilters), settings.Format, inputFilters)
        {
            Shader = settings.Shader;
            LinearSampling = settings.PerTextureLinearSampling
                .Concat(Enumerable.Repeat(settings.LinearSampling, inputFilters.Length - settings.PerTextureLinearSampling.Length))
                .ToArray();

            Args = settings.Arguments;
            this.AddLabel(settings.Name, 50);
        }

        protected abstract void SetTextureConstant(int index, IBaseTexture texture, bool linearSampling);
        protected abstract void SetConstant(string identifier, Vector4 value);
        protected abstract void Render(T shader);

        protected virtual void LoadCustomConstants()
        {
            // override to load custom inputs such as a weights buffer
        }

        protected T Shader { get; private set; }
        protected bool[] LinearSampling { get; private set; }
        protected ArgumentList Args { get; private set; }

        protected void LoadTextureConstants(IList<IBaseTexture> inputs)
        {
            var index = 0;
            foreach (var input in inputs)
            {
                SetTextureConstant(index, input, LinearSampling[index]);
                LoadSizeConstant(index.ToString(), input.GetSize());
                index++;
            }

            /* Add Output Size */
            LoadSizeConstant("Output", Output.Size);
        }

        private void LoadSizeConstant(string identifier, TextureSize size)
        {
            if (size.Is3D)
                SetConstant(string.Format("size3d{0}", identifier), new Vector4(size.Width, size.Height, size.Depth, 0));
            else
                SetConstant(string.Format("size{0}", identifier), new Vector4(size.Width, size.Height, 1.0f / size.Width, 1.0f / size.Height));
        }

        protected virtual void LoadInputs(IList<IBaseTexture> inputs)
        {
            LoadTextureConstants(inputs);
            LoadShaderConstants();
            LoadCustomConstants();
        }

        protected void LoadShaderConstants()
        {
            foreach (var argument in Args)
                SetConstant(argument.Key, argument.Value);
        }

        protected override void Render(IList<ITextureOutput<IBaseTexture>> inputs)
        {
            LoadInputs(inputs.Select(x => x.Texture).ToList());
            Render(Shader);
        }
    }

    public class ShaderFilter : GenericShaderFilter<IShader>
    {
        public ShaderFilter(IShaderFilterSettings<IShader> settings, params IBaseTextureFilter[] inputFilters)
            : base(settings, inputFilters)
        {
        }

        public ShaderFilter(IShader shader, params IBaseTextureFilter[] inputFilters)
            : base(shader, inputFilters)
        {
        }

        protected int Counter { get; private set; }

        protected void LoadLegacyConstants()
        {
            // Legacy constants 
            var output = Output.Texture;
            Shader.SetConstant(0, new Vector4(output.Width, output.Height, Counter++ & 0x7fffff, Renderer.FrameTimeStampMicrosec / 1000000.0f), false);
            Shader.SetConstant(1, new Vector4(1.0f / output.Width, 1.0f / output.Height, 0, 0), false);
        }

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

        protected override void LoadCustomConstants()
        {
            LoadLegacyConstants();
        }

        protected override void Render(IShader shader)
        {
            Renderer.Render(Target.Texture, shader);
        }
    }

    public class Shader11Filter : GenericShaderFilter<IShader11>
    {
        private readonly IList<IDisposable> m_Buffers = new List<IDisposable>();

        public Shader11Filter(IShaderFilterSettings<IShader11> settings, params IBaseTextureFilter[] inputFilters)
            : base(settings, inputFilters)
        {
        }

        public Shader11Filter(IShader11 shader, params IBaseTextureFilter[] inputFilters)
            : base(shader, inputFilters)
        {
        }

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

        protected override void Render(IShader11 shader)
        {
            Renderer.Render(Target.Texture, shader);

            DisposeHelper.DisposeElements(m_Buffers);
            m_Buffers.Clear();
        }
    }

    public class DirectComputeFilter : Shader11Filter
    {
        public DirectComputeFilter(IShaderFilterSettings<IShader11> settings, int threadGroupX, int threadGroupY,
            int threadGroupZ, params IBaseTextureFilter[] inputFilters)
            : base(settings, inputFilters)
        {
            ThreadGroupX = threadGroupX;
            ThreadGroupY = threadGroupY;
            ThreadGroupZ = threadGroupZ;
        }

        public DirectComputeFilter(IShader11 shader, int threadGroupX, int threadGroupY, int threadGroupZ,
            params IBaseTextureFilter[] inputFilters)
            : base(shader, inputFilters)
        {
            ThreadGroupX = threadGroupX;
            ThreadGroupY = threadGroupY;
            ThreadGroupZ = threadGroupZ;
        }

        protected override void Render(IShader11 shader)
        {
            Renderer.Compute(Target.Texture, shader, ThreadGroupX, ThreadGroupY, ThreadGroupZ);
        }

        public int ThreadGroupX { get; private set; }
        public int ThreadGroupY { get; private set; }
        public int ThreadGroupZ { get; private set; }
    }

    public class ClKernelFilter : GenericShaderFilter<IKernel>
    {
        public ClKernelFilter(IShaderFilterSettings<IKernel> settings, int[] globalWorkSizes, params IBaseTextureFilter[] inputFilters)
            : base(settings, inputFilters)
        {
            GlobalWorkSizes = globalWorkSizes;
            LocalWorkSizes = null;
        }

        public ClKernelFilter(IKernel shader, int[] globalWorkSizes, params IBaseTextureFilter[] inputFilters)
            : base(shader, inputFilters)
        {
            GlobalWorkSizes = globalWorkSizes;
            LocalWorkSizes = null;
        }

        public ClKernelFilter(IShaderFilterSettings<IKernel> settings, int[] globalWorkSizes, int[] localWorkSizes, params IBaseTextureFilter[] inputFilters)
            : base(settings, inputFilters)
        {
            GlobalWorkSizes = globalWorkSizes;
            LocalWorkSizes = localWorkSizes;
        }

        public ClKernelFilter(IKernel shader, int[] globalWorkSizes, int[] localWorkSizes, params IBaseTextureFilter[] inputFilters)
            : base(shader, inputFilters)
        {
            GlobalWorkSizes = globalWorkSizes;
            LocalWorkSizes = localWorkSizes;
        }

        protected override void SetTextureConstant(int index, IBaseTexture texture, bool linearSampling)
        {
            var tex = texture as ITexture2D;
            if (tex == null)
                throw new NotSupportedException("Only 2D textures are supported in OpenCL");

            Shader.SetInputTextureArg(index, tex, false);
        }

        protected override void SetConstant(string identifier, Vector4 value) { } // Can't be implemented at this time

        protected override void Render(IKernel shader)
        {
            Renderer.RunClKernel(shader, GlobalWorkSizes, LocalWorkSizes);
        }

        public int[] GlobalWorkSizes { get; private set; }
        public int[] LocalWorkSizes { get; private set; }
    }
}
