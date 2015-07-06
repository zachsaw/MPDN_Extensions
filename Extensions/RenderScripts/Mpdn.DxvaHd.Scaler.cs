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
using System.Drawing;
using System.Linq;
using Mpdn.DxvaHd;
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.Framework.Exceptions;
using Mpdn.Extensions.Framework.RenderChain;
using Mpdn.RenderScript;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Mpdn.DxvaHd
    {
        public class DxvaHdScaler : RenderChain
        {
            #region Settings

            public DxvaHdScaler()
            {
                Quality = DxvaHdQuality.Quality;
                YuvMode = true;
            }

            public DxvaHdQuality Quality { get; set; }
            public bool YuvMode { get; set; }

            #endregion

            #region Filter Classes

            private class DxvaHdResizeFilter : ResizeFilter
            {
                private readonly IDxvaHd m_DxvaHd;

                public DxvaHdResizeFilter(IDxvaHd dxvaHd, IFilter inputFilter)
                    : base(inputFilter)
                {
                    m_DxvaHd = dxvaHd;
                    SetSize(Renderer.TargetSize);
                }

                protected override IFilter<ITexture2D> Optimize()
                {
                    return this;
                }

                protected override void Render(IList<IBaseTexture> inputs)
                {
                    var texture = inputs.OfType<ITexture2D>().SingleOrDefault();
                    if (texture == null)
                        return;

                    if (texture.Format != TextureFormat.Unorm8 || OutputTarget.Format != TextureFormat.Unorm8)
                    {
                        throw new RenderScriptException("Input and output formats must be Unorm8");
                    }

                    m_DxvaHd.Render(texture, OutputTarget);
                }
            }

            #endregion

            private IDxvaHd m_DxvaHd;
            private ITargetTexture m_TextureUnorm8;

            protected override string ShaderPath
            {
                get { return "DxvaHdScaler"; }
            }

            public override void Reset()
            {
                Cleanup();

                base.Reset();
            }

            public override IFilter CreateFilter(IFilter sourceFilter)
            {
                Cleanup();

                if (sourceFilter.OutputSize == Renderer.TargetSize)
                    return sourceFilter;

                try
                {
                    m_DxvaHd = Renderer.CreateDxvaHd((Size) sourceFilter.OutputSize, TextureFormat.Unorm8,
                        Renderer.TargetSize, TextureFormat.Unorm8, Quality);
                }
                catch (DxvaHdException)
                {
                    // DXVA HD not available; fallback
                    Renderer.FallbackOccurred = true;
                    return sourceFilter;
                }

                var input = YuvMode ? sourceFilter.ConvertToYuv() : sourceFilter;

                if (sourceFilter.OutputFormat != TextureFormat.Unorm8)
                {
                    // Convert input to Unorm8 (and unforunately murdering quality at the same time)
                    var copy = CompileShader("Copy.hlsl").Configure(linearSampling: false, format: TextureFormat.Unorm8);
                    input = new ShaderFilter(copy, input);
                }

                var result = new DxvaHdResizeFilter(m_DxvaHd, input);

                return YuvMode ? result.ConvertToRgb() : result;
            }

            private void Cleanup()
            {
                DisposeHelper.Dispose(ref m_DxvaHd);
                DisposeHelper.Dispose(ref m_TextureUnorm8);
            }
        }

        public class DxvaHdScalerScaler : RenderChainUi<DxvaHdScaler, DxvaHdScalerConfigDialog>
        {
            protected override string ConfigFileName
            {
                get { return "Mpdn.DxvaHdScaler"; }
            }

            public override string Category
            {
                get { return "Scaling"; }
            }

            public override ExtensionUiDescriptor Descriptor
            {
                get
                {
                    return new ExtensionUiDescriptor
                    {
                        Guid = new Guid("FF0351E6-9320-4260-8982-3DFD3A4223A5"),
                        Name = "DXVA HD Scaler",
                        Description = "Scales image to target size using DXVA HD scaler (8-bit)"
                    };
                }
            }
        }
    }
}
