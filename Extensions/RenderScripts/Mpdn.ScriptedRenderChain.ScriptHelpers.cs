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
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Windows.Forms;
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.Framework.Config;
using Mpdn.RenderScript;
using SharpDX;
using Point = System.Drawing.Point;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Mpdn.ScriptedRenderChain
    {
        public static class Debug
        {
            public static void Output(object text)
            {
                Trace.WriteLine(text.ToString());
            }
        }

        public class InternalHostFunctions
        {
            public void AssignProp(object obj, string propName, dynamic value)
            {
                if (obj == null)
                {
                    throw new ArgumentNullException("obj");
                }

                var propInfo = obj.GetType().GetProperty(propName);
                if (propInfo == null)
                {
                    throw new ArgumentException(string.Format("Invalid property name '{0}'", propName), "propName");
                }

                var propType = propInfo.PropertyType;
                if (propType.IsArray)
                {
                    var length = value.length;
                    var arr = Array.CreateInstance(propType.GetElementType(), length);
                    for (int i = 0; i < length; i++)
                    {
                        arr[i] = value[i];
                    }
                    propInfo.SetValue(obj, arr, null);
                }
                else
                {
                    propInfo.SetValue(obj, value, null);
                }
            }
        }

        public class Host
        {
            public string ExePath
            {
                get { return PathHelper.GetDirectoryName(Application.ExecutablePath); }
            }

            public string ExeName
            {
                get { return Path.GetFileName(Application.ExecutablePath); }
            }

            public string StartupPath
            {
                get { return Application.StartupPath; }
            }

            public string ConfigFile
            {
                get { return PlayerControl.ConfigRootPath; }
            }

            public string ConfigFilePath
            {
                get { return PathHelper.GetDirectoryName(PlayerControl.ConfigRootPath); }
            }

            public string Version
            {
                get { return Application.ProductVersion; }
            }

            public string Name
            {
                get { return Application.ProductName; }
            }
        }

        public class Script
        {
            private static IEnumerable<IRenderChainUi> s_RenderScripts;

            public dynamic Load(string name)
            {
                var chainUi = GetRenderScripts().FirstOrDefault(script => script.Descriptor.Name == name);
                if (chainUi == null)
                {
                    throw new ArgumentException(string.Format("script.Load() error: Script '{0}' not found", name));
                }

                ((IConfigLoadable) chainUi).LoadConfig();
                return chainUi.Chain;
            }

            public dynamic LoadByClassName(string className)
            {
                var chainUi = GetRenderScripts().FirstOrDefault(script => script.Chain.GetType().Name == className);
                if (chainUi == null)
                {
                    throw new ArgumentException(
                        string.Format("script.Load() error: Script with class name '{0}' not found", className));
                }

                ((IConfigLoadable) chainUi).LoadConfig();
                return chainUi.Chain;
            }

            private static IEnumerable<IRenderChainUi> GetRenderScripts()
            {
                return s_RenderScripts ??
                       (s_RenderScripts = PlayerControl.RenderScripts
                           .Where(script => script is IRenderChainUi && script is IConfigLoadable)
                           .Select(x => (x as IRenderChainUi).CreateNew()));
            }
        }

        public class Clip
        {
            private readonly RenderChain m_Chain;
            public IFilter Filter { get; private set; }

            public string FileName
            {
                get { return Renderer.VideoFileName; }
            }

            public bool Interlaced
            {
                get { return Renderer.InterlaceFlags.HasFlag(InterlaceFlags.IsInterlaced); }
            }

            public bool NeedsUpscaling
            {
                get { return m_Chain.IsUpscalingFrom(Filter); }
            }

            public bool NeedsDownscaling
            {
                get { return m_Chain.IsDownscalingFrom(Filter); }
            }

            public Size TargetSize
            {
                get { return Renderer.TargetSize; }
            }

            public Size SourceSize
            {
                get { return Renderer.VideoSize; }
            }

            public Size LumaSize
            {
                get { return Renderer.LumaSize; }
            }

            public Size ChromaSize
            {
                get { return Renderer.ChromaSize; }
            }

            public Vector2 ChromaOffset
            {
                get { return Renderer.ChromaOffset; }
            }

            public Point AspectRatio
            {
                get { return Renderer.AspectRatio; }
            }

            public YuvColorimetric Colorimetric
            {
                get { return Renderer.Colorimetric; }
            }

            public FrameBufferInputFormat InputFormat
            {
                get { return Renderer.InputFormat; }
            }

            public double FrameRateHz
            {
                get { return Renderer.FrameRateHz; }
            }

            public Clip(RenderChain chain, IFilter input)
            {
                m_Chain = chain;
                Filter = input;
            }

            public Clip Add(RenderChain filter)
            {
                if (filter == null)
                {
                    throw new ArgumentNullException("filter");
                }
                Filter += filter;
                return this;
            }

            public Clip Apply(RenderChain filter)
            {
                return Add(filter);
            }
        }

        public static class Helpers
        {
            public static string SubstringIdx(this string self, int startIndex, int endIndex)
            {
                var length = endIndex - startIndex;
                if (length < 0)
                {
                    length = 0;
                }
                return self.Substring(startIndex, length);
            }

            public static string DefaultScript
            {
                get
                {
                    return
                        @"// Example render script

// Scale chroma first (this bypasses MPDN's chroma scaler)
BicubicChroma( Preset = Presets.MitchellNetravali )

// Apply some filtering pixel shaders
ImageProcessor( ShaderFileNames = [""SweetFX\\Bloom.hlsl"", ""SweetFX\\LiftGammaGain.hlsl""] )

// Use NEDI once only.
// Note: To use NEDI as many times as required to get the image past target size,
//       change the following *if* to *while*
if (input.NeedsUpscaling)
{
    Nedi( AlwaysDoubleImage = true )
}

if (input.NeedsDownscaling)
{
    // Use linear light for downscaling
    ImageProcessor( ShaderFileNames = [""ConvertToLinearLight.hlsl""] )
    Resizer( ResizerOption = ResizerOption.TargetSize100Percent )
    ImageProcessor( ShaderFileNames = [""ConvertToGammaLight.hlsl""] )
}

if (input.SourceSize.Width < 1920)
{
    // Sharpen only if video isn't full HD
    // Or if you have FineSharp installed, replace the following line with it
    ImageProcessor( ShaderFileNames = [""SweetFX\\LumaSharpen.hlsl""] )
}
";
                }
            }
        }
    }
}
