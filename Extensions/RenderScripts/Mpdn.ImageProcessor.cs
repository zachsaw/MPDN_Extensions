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
using System.IO;
using System.Linq;
using Mpdn.Extensions.Framework.RenderChain;
using Mpdn.Extensions.Framework.RenderChain.Shaders;
using Mpdn.RenderScript;
using SharpDX;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Mpdn.ImageProcessor
    {
        public class ImageProcessor : RenderChain
        {
            #region Settings

            private string[] m_ShaderFileNames;

            public string[] ShaderFileNames
            {
                get { return m_ShaderFileNames ?? (m_ShaderFileNames = new string[0]); }
                set { m_ShaderFileNames = value; }
            }

            public bool CompatibilityMode { get; set; }

            public bool ProcessInYUV { get; set; }

            public ImageProcessor()
            {
                CompatibilityMode = true;
            }

            #endregion

            protected override string ShaderPath
            {
                get { return "ImageProcessingShaders"; }
            }

            public string FullShaderPath
            {
                get { return ShaderDataFilePath; }
            }

            public override string Description
            {
                get
                {
                    var count = ShaderFileNames.Count();
                    if (count == 0) return string.Empty;
                    var result = count == 1
                        ? Path.GetFileNameWithoutExtension(ShaderFileNames.First())
                        : string.Format("{0}..{1}", Path.GetFileNameWithoutExtension(ShaderFileNames.First()),
                            Path.GetFileNameWithoutExtension(ShaderFileNames.Last()));
                    return string.Format("ImageProc('{0}')", result);
                }
            }

            protected override ITextureFilter CreateFilter(ITextureFilter input)
            {
                if (ProcessInYUV)
                    input = input.ConvertToYuv();

                var output = ShaderFileNames.Aggregate(input,
                    (current, filename) => current.Apply(GetShader(filename)));

                return ProcessInYUV
                    ? output.ConvertToRgb()
                    : output;
            }

            private class LegacyShader : Shader
            {
                public LegacyShader(IShaderDefinition<IShader> config) : base(config) { }
                public LegacyShader(Shader config) : base(config) { }

                private class LegacyHandle : ShaderHandle
                {
                    private int m_Counter;

                    public LegacyHandle(IShaderParameters parameters, IShaderDefinition<IShader> definition) 
                        : base(parameters, definition)
                    { }

                    protected override void LoadArguments(IList<IBaseTexture> inputs, ITargetTexture output)
                    {
                        base.LoadArguments(inputs, output);

                        // Load Legacy Constants
                        Shader.SetConstant(0, new Vector4(output.Width, output.Height, m_Counter++ & 0x7fffff, Renderer.FrameTimeStampMicrosec / 1000000.0f), false);
                        Shader.SetConstant(1, new Vector4(1.0f / output.Width, 1.0f / output.Height, 0, 0), false);
                    }
                }

                public override IShaderHandle GetHandle()
                {
                    return new LegacyHandle(this, Definition);
                }
            }

            private Shader GetShader(string filename)
            {
                var shader = (CompatibilityMode)
                    ? new LegacyShader(FromFile(filename))
                    : new Shader(FromFile(filename));
                if (CompatibilityMode)
                    shader.Format = (Renderer.RenderQuality == RenderQuality.MaxQuality)
                        ? TextureFormat.Float32
                        : TextureFormat.Float16;
                return shader;
            }
        }

        public class ImageProcessorScript : RenderChainUi<ImageProcessor, ImageProcessorConfigDialog>
        {
            protected override string ConfigFileName
            {
                get { return "Mpdn.ImageProcessor"; }
            }

            public override string Category
            {
                get { return "Processing"; }
            }

            public override ExtensionUiDescriptor Descriptor
            {
                get
                {
                    return new ExtensionUiDescriptor
                    {
                        Guid = new Guid("50CA262F-65B6-4A0F-A8B5-5E25B6A18217"),
                        Name = "Image Processor",
                        Description = GetDescription(),
                    };
                }
            }

            private string GetDescription()
            {
                return Settings.ShaderFileNames.Length == 0
                    ? "Pixel shader pre-/post-processing filter"
                    : string.Join(" ➔ ", Settings.ShaderFileNames);
            }
        }
    }
}
