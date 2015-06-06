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

using System.IO;
using Mpdn.Extensions.Framework;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Mpdn.ScriptedRenderChain
    {
        public static class Helpers
        {
            public static string DefaultScriptFileName
            {
                get { return Path.Combine(PathHelper.GetDirectoryName(PlayerControl.ConfigRootPath), "DefaultScript.rs"); }
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
