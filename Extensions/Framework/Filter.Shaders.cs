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
using System.Collections.Generic;
using System.Linq;
using Mpdn.OpenCl;
using Mpdn.RenderScript;
using SharpDX;
using TransformFunc = System.Func<Mpdn.Extensions.Framework.TextureSize, Mpdn.Extensions.Framework.TextureSize>;
using IBaseFilter = Mpdn.Extensions.Framework.IFilter<Mpdn.IBaseTexture>;

namespace Mpdn.Extensions.Framework
{
    public class ShaderFilterSettings<T>
    {
        public T Shader;
        public bool LinearSampling;
        public bool[] PerTextureLinearSampling = new bool[0];
        public TransformFunc Transform = (s => s);
        public TextureFormat Format = Renderer.RenderQuality.GetTextureFormat();
        public int SizeIndex;
        public float[] Args = new float[0];

        public ShaderFilterSettings(T shader)
        {
            Shader = shader;
        }

        public static implicit operator ShaderFilterSettings<T>(T shader)
        {
            return new ShaderFilterSettings<T>(shader);
        }

        public ShaderFilterSettings<T> Configure(bool? linearSampling = null, 
            float[] arguments = null, TransformFunc transform = null, int? sizeIndex = null, TextureFormat? format = null, 
            bool[] perTextureLinearSampling = null)
        {
            return new ShaderFilterSettings<T>(Shader)
            {
                Transform = transform ?? Transform,
                LinearSampling = linearSampling ?? LinearSampling,
                Format = format ?? Format,
                SizeIndex = sizeIndex ?? SizeIndex,
                Args = arguments ?? Args,
                PerTextureLinearSampling = perTextureLinearSampling ?? PerTextureLinearSampling
            };
        }
    }

    public static class ShaderFilterHelper
    {
        public static ShaderFilterSettings<IShader> Configure(this IShader shader, bool? linearSampling = null,
            float[] arguments = null, TransformFunc transform = null, int? sizeIndex = null,
            TextureFormat? format = null, bool[] perTextureLinearSampling = null)
        {
            return new ShaderFilterSettings<IShader>(shader).Configure(linearSampling, arguments, transform, sizeIndex,
                format, perTextureLinearSampling);
        }

        public static ShaderFilterSettings<IShader11> Configure(this IShader11 shader, bool? linearSampling = null,
            float[] arguments = null, TransformFunc transform = null, int? sizeIndex = null,
            TextureFormat? format = null, bool[] perTextureLinearSampling = null)
        {
            return new ShaderFilterSettings<IShader11>(shader).Configure(linearSampling, arguments, transform, sizeIndex,
                format, perTextureLinearSampling);
        }

        public static ShaderFilterSettings<IKernel> Configure(this IKernel kernel, bool? linearSampling = null,
            float[] arguments = null, TransformFunc transform = null, int? sizeIndex = null,
            TextureFormat? format = null, bool[] perTextureLinearSampling = null)
        {
            return new ShaderFilterSettings<IKernel>(kernel).Configure(linearSampling, arguments, transform, sizeIndex,
                format, perTextureLinearSampling);
        }
    }

    public abstract class GenericShaderFilter<T> : Filter where T : class
    {
        protected GenericShaderFilter(T shader, params IBaseFilter[] inputFilters)
            : this((ShaderFilterSettings<T>) shader, inputFilters)
        {
        }

        protected GenericShaderFilter(ShaderFilterSettings<T> settings, params IBaseFilter[] inputFilters)
            : base(inputFilters)
        {
            Shader = settings.Shader;
            LinearSampling = settings.PerTextureLinearSampling.Length > 0
                ? settings.PerTextureLinearSampling
                : Enumerable.Repeat(settings.LinearSampling, inputFilters.Length).ToArray();
            Transform = settings.Transform;
            Format = settings.Format;
            SizeIndex = settings.SizeIndex;

            if (SizeIndex < 0 || SizeIndex >= inputFilters.Length || inputFilters[SizeIndex] == null)
            {
                throw new IndexOutOfRangeException(String.Format("No valid input filter at index {0}", SizeIndex));
            }

            var arguments = settings.Args ?? new float[0];
            Args = new float[4*((arguments.Length + 3)/4)];
            arguments.CopyTo(Args, 0);
        }

        protected T Shader { get; private set; }
        protected bool[] LinearSampling { get; private set; }
        protected TransformFunc Transform { get; private set; }
        protected TextureFormat Format { get; private set; }
        protected int SizeIndex { get; private set; }
        protected float[] Args { get; private set; }

        public override TextureSize OutputSize
        {
            get { return Transform(InputFilters[SizeIndex].OutputSize); }
        }

        public override TextureFormat OutputFormat
        {
            get { return Format; }
        }

        protected abstract void LoadInputs(IList<IBaseTexture> inputs);
        protected abstract void Render(T shader);

        protected override void Render(IList<IBaseTexture> inputs)
        {
            LoadInputs(inputs);
            Render(Shader);
        }
    }

    public class ShaderFilter : GenericShaderFilter<IShader>
    {
        public ShaderFilter(ShaderFilterSettings<IShader> settings, params IBaseFilter[] inputFilters)
            : base(settings, inputFilters)
        {
        }

        public ShaderFilter(IShader shader, params IBaseFilter[] inputFilters)
            : base(shader, inputFilters)
        {
        }

        protected int Counter { get; private set; }

        protected override void LoadInputs(IList<IBaseTexture> inputs)
        {
            var i = 0;
            foreach (var input in inputs)
            {
                if (input is ITexture2D)
                {
                    var tex = (ITexture2D) input;
                    Shader.SetTextureConstant(i, tex, LinearSampling[i], false);
                    Shader.SetConstant(String.Format("size{0}", i),
                        new Vector4(tex.Width, tex.Height, 1.0f/tex.Width, 1.0f/tex.Height), false);
                }
                else
                {
                    var tex = (ITexture3D) input;
                    Shader.SetTextureConstant(i, tex, LinearSampling[i], false);
                    Shader.SetConstant(String.Format("size3d{0}", i),
                        new Vector4(tex.Width, tex.Height, tex.Depth, 0), false);
                }
                i++;
            }

            for (i = 0; 4*i < Args.Length; i++)
            {
                Shader.SetConstant(String.Format("args{0}", i),
                    new Vector4(Args[4*i], Args[4*i + 1], Args[4*i + 2], Args[4*i + 3]), false);
            }

            // Legacy constants 
            var output = OutputTarget;
            Shader.SetConstant(0, new Vector4(output.Width, output.Height, Counter++ & 0x7fffff, Renderer.FrameTimeStampMicrosec / 1000000.0f),
                false);
            Shader.SetConstant(1, new Vector4(1.0f/output.Width, 1.0f/output.Height, 0, 0), false);
        }

        protected override void Render(IShader shader)
        {
            Renderer.Render(OutputTarget, shader);
        }
    }

    public class Shader11Filter : GenericShaderFilter<IShader11>
    {
        public Shader11Filter(ShaderFilterSettings<IShader11> settings, params IBaseFilter[] inputFilters)
            : base(settings, inputFilters)
        {
        }

        public Shader11Filter(IShader11 shader, params IBaseFilter[] inputFilters)
            : base(shader, inputFilters)
        {
        }

        protected int Counter { get; private set; }

        protected override void LoadInputs(IList<IBaseTexture> inputs)
        {
            var i = 0;
            foreach (var input in inputs)
            {
                if (input is ITexture2D)
                {
                    var tex = (ITexture2D) input;
                    Shader.SetTextureConstant(i, tex, LinearSampling[i], false);
                    Shader.SetConstantBuffer(String.Format("size{0}", i),
                        new Vector4(tex.Width, tex.Height, 1.0f/tex.Width, 1.0f/tex.Height), false);
                }
                else
                {
                    var tex = (ITexture3D) input;
                    Shader.SetTextureConstant(i, tex, LinearSampling[i], false);
                    Shader.SetConstantBuffer(String.Format("size3d{0}", i),
                        new Vector4(tex.Width, tex.Height, tex.Depth, 0), false);
                }
                i++;
            }

            for (i = 0; 4*i < Args.Length; i++)
            {
                Shader.SetConstantBuffer(String.Format("args{0}", i),
                    new Vector4(Args[4*i], Args[4*i + 1], Args[4*i + 2], Args[4*i + 3]), false);
            }

            // Legacy constants 
            var output = OutputTarget;
            Shader.SetConstantBuffer(0, new Vector4(output.Width, output.Height, Counter++ & 0x7fffff, 
                Renderer.FrameTimeStampMicrosec / 1000000.0f), false);
        }

        protected override void Render(IShader11 shader)
        {
            Renderer.Render(OutputTarget, shader);
        }
    }

    public class DirectComputeFilter : Shader11Filter
    {
        public DirectComputeFilter(ShaderFilterSettings<IShader11> settings, int threadGroupX, int threadGroupY,
            int threadGroupZ, params IBaseFilter[] inputFilters)
            : base(settings, inputFilters)
        {
            ThreadGroupX = threadGroupX;
            ThreadGroupY = threadGroupY;
            ThreadGroupZ = threadGroupZ;
        }

        public DirectComputeFilter(IShader11 shader, int threadGroupX, int threadGroupY, int threadGroupZ,
            params IBaseFilter[] inputFilters)
            : base(shader, inputFilters)
        {
            ThreadGroupX = threadGroupX;
            ThreadGroupY = threadGroupY;
            ThreadGroupZ = threadGroupZ;
        }

        protected override void Render(IShader11 shader)
        {
            Renderer.Compute(OutputTarget, shader, ThreadGroupX, ThreadGroupY, ThreadGroupZ);
        }

        public int ThreadGroupX { get; private set; }
        public int ThreadGroupY { get; private set; }
        public int ThreadGroupZ { get; private set; }
    }

    public class ClKernelFilter : GenericShaderFilter<IKernel>
    {
        public ClKernelFilter(ShaderFilterSettings<IKernel> settings, int[] globalWorkSizes, params IBaseFilter[] inputFilters)
            : base(settings, inputFilters)
        {
            GlobalWorkSizes = globalWorkSizes;
            LocalWorkSizes = null;
        }

        public ClKernelFilter(IKernel shader, int[] globalWorkSizes, params IBaseFilter[] inputFilters)
            : base(shader, inputFilters)
        {
            GlobalWorkSizes = globalWorkSizes;
            LocalWorkSizes = null;
        }

        public ClKernelFilter(ShaderFilterSettings<IKernel> settings, int[] globalWorkSizes, int[] localWorkSizes, params IBaseFilter[] inputFilters)
            : base(settings, inputFilters)
        {
            GlobalWorkSizes = globalWorkSizes;
            LocalWorkSizes = localWorkSizes;
        }

        public ClKernelFilter(IKernel shader, int[] globalWorkSizes, int[] localWorkSizes, params IBaseFilter[] inputFilters)
            : base(shader, inputFilters)
        {
            GlobalWorkSizes = globalWorkSizes;
            LocalWorkSizes = localWorkSizes;
        }

        protected virtual void LoadCustomInputs()
        {
            // override to load custom OpenCL inputs such as a weights buffer
        }

        protected override void LoadInputs(IList<IBaseTexture> inputs)
        {
            Shader.SetOutputTextureArg(0, OutputTarget); // Note: MPDN only supports one output texture per kernel

            var i = 1;
            foreach (var input in inputs)
            {
                if (input is ITexture2D)
                {
                    var tex = (ITexture2D) input;
                    Shader.SetInputTextureArg(i, tex, false);
                }
                else
                {
                    throw new NotSupportedException("Only 2D textures are supported in OpenCL");
                }
                i++;
            }

            foreach (var v in Args)
            {
                Shader.SetArg(i++, v, false);
            }

            LoadCustomInputs();
        }

        protected override void Render(IKernel shader)
        {
            Renderer.RunClKernel(shader, GlobalWorkSizes, LocalWorkSizes);
        }

        public int[] GlobalWorkSizes { get; private set; }
        public int[] LocalWorkSizes { get; private set; }
    }
}
