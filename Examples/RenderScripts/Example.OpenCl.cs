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
using Mpdn.Extensions.Framework.Filter;
using Mpdn.Extensions.Framework.RenderChain;
using Mpdn.Extensions.RenderScripts.Mpdn.Resizer;
using Mpdn.RenderScript;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Example
    {
        // Pass color as arguments
        public class OpenCl : RenderChain
        {
            protected override string ShaderPath
            {
                get { return "Examples"; }
            }

            public override IFilter CreateFilter(IFilter sourceFilter)
            {
                if (!Renderer.IsOpenClAvail || Renderer.RenderQuality == RenderQuality.MaxPerformance)
                    return new NullFilter(); // display blank screen on purpose

                // get MPDN to scale image to target size first
                sourceFilter += new Resizer { ResizerOption = ResizerOption.TargetSize100Percent };

                // apply our blue tint
                var blueTint = CompileClKernel("BlueTint.cl", "BlueTint")
                    .Configure(arguments: new[] {0.25f, 0.5f, 0.75f});

                var outputSize = sourceFilter.OutputSize;
                return new ClKernelFilter(blueTint, new[] {outputSize.Width, outputSize.Height}, sourceFilter);
            }
        }

/*
        // Pass color as OpenCL buffer (remember to change BlueTint.cl to match)
        public class OpenCl : RenderChain
        {
            private IDisposable m_Buffer;

            private class MyKernelFilter : ClKernelFilter
            {
                private readonly IDisposable m_Buffer;

                public MyKernelFilter(ShaderFilterSettings<IKernel> settings, IDisposable buffer, int[] workSizes, 
                    params IFilter<IBaseTexture>[] inputFilters) 
                    : base(settings, workSizes, inputFilters)
                {
                    m_Buffer = buffer;
                }

                protected override void LoadCustomInputs()
                {
                    Shader.SetBufferArg(2, m_Buffer);
                }
            }

            public override void RenderScriptDisposed()
            {
                DisposeHelper.Dispose(ref m_Buffer);

                base.RenderScriptDisposed();
            }

            protected override string ShaderPath
            {
                get { return "Examples"; }
            }

            public override IFilter CreateFilter(IFilter sourceFilter)
            {
                if (!Renderer.IsOpenClAvail || Renderer.RenderQuality == RenderQuality.MaxPerformance)
                    return new NullFilter(); // display blank screen on purpose

                m_Buffer = Renderer.CreateClBuffer(new[] {0.25f, 0.5f, 0.75f});

                // get MPDN to scale image to target size first
                sourceFilter += new Resizer { ResizerOption = ResizerOption.TargetSize100Percent };

                // apply our blue tint
                var blueTint = CompileClKernel("BlueTint.cl", "BlueTint").Configure();

                var outputSize = sourceFilter.OutputSize;
                return new MyKernelFilter(blueTint, m_Buffer, new[] {outputSize.Width, outputSize.Height}, sourceFilter);
            }
        }
*/

/*
        // Pass color as SharpDX.Vector4 (remember to change BlueTint.cl to match)
        public class OpenCl : RenderChain
        {
            private class MyKernelFilter : ClKernelFilter
            {
                private readonly Vector4 m_Color;

                public MyKernelFilter(ShaderFilterSettings<IKernel> settings, Vector4 color, int[] workSizes,
                    params IFilter<IBaseTexture>[] inputFilters)
                    : base(settings, workSizes, inputFilters)
                {
                    m_Color = color;
                }

                protected override void LoadCustomInputs()
                {
                    Shader.SetArg(2, m_Color);
                }
            }

            protected override string ShaderPath
            {
                get { return "Examples"; }
            }

            public override IFilter CreateFilter(IFilter sourceFilter)
            {
                if (!Renderer.IsOpenClAvail || Renderer.RenderQuality == RenderQuality.MaxPerformance)
                    return new NullFilter(); // display blank screen on purpose

                // get MPDN to scale image to target size first
                sourceFilter += new Resizer {ResizerOption = ResizerOption.TargetSize100Percent};

                // apply our blue tint
                var blueTint = CompileClKernel("BlueTint.cl", "BlueTint").Configure();

                var outputSize = sourceFilter.OutputSize;
                return new MyKernelFilter(blueTint, new Vector4(0.25f, 0.5f, 0.75f, 0.0f),
                    new[] {outputSize.Width, outputSize.Height}, sourceFilter);
            }
        }
*/

        public class OpenClExample : RenderChainUi<OpenCl>
        {
            public override ExtensionUiDescriptor Descriptor
            {
                get
                {
                    return new ExtensionUiDescriptor
                    {
                        Name = "OpenCL Blue Tint Example",
                        Description = "(Example) Applies a blue tint over the image using OpenCL",
                        Guid = new Guid("563CCFB4-C67E-438B-856D-6CD3763BE19E"),
                        Copyright = "" // Optional field
                    };
                }
            }

            public override string Category
            {
                get { return "Example"; }
            }
        }
    }
}
