//
// NOTE: MPDN needs to be restarted for any changes you have made in this file to take effect!
//

using System.Collections.Generic;
using System.Drawing;
using Mpdn.RenderScript.Mpdn.ImageProcessor;
using Mpdn.RenderScript.Mpdn.Resizer;
using Mpdn.RenderScript.Shiandow.Chroma;
using Mpdn.RenderScript.Shiandow.Nedi;

namespace Mpdn.RenderScript
{
    namespace MyOwnUniqueNameSpace // e.g. Replace with your user name
    {
        public class MyRenderScript : CustomRenderScriptChain
        {
            private readonly string[] PreResizeShaderfiles = { @"SweetFX\Bloom.hlsl", /* add more files here (separate with comma) ... */ };
            private readonly string[] PostResizeShaderfiles = { @"SweetFX\LumaSharpen.hlsl", /* add more files here (separate with comma) ... */ };

            private RenderScript ScaleChroma, Nedi, PreProcess, PostProcess, Deinterlace, ToLinear, ToGamma, ResizeToTarget;

            protected override RenderScript[] CreateScripts()
            {
                return new[]
                {
                    ScaleChroma = ChromaScaler.Create(preset: Presets.MitchellNetravali),
                    Nedi = NediScaler.Create(forced: true),
                    PreProcess = ImageProcessor.Create(PreResizeShaderfiles),
                    PostProcess = ImageProcessor.Create(PostResizeShaderfiles),
                    Deinterlace = ImageProcessor.Create(new[] {@"MPC-HC\Deinterlace (blend).hlsl"}),
                    ToLinear = ImageProcessor.Create(new[] {@"ConvertToLinearLight.hlsl"}),
                    ToGamma = ImageProcessor.Create(new[] {@"ConvertToGammaLight.hlsl"}),
                    ResizeToTarget = Resizer.Create(option: ResizerOption.TargetSize100Percent),
                    // Add more scripts here ...
                };
            }

            protected override RenderScript[] GetScriptChain()
            {
                var result = new List<RenderScript>();

                // Scale chroma first (this bypasses MPDN's chroma scaler)
                result.Add(ScaleChroma);

                if (Renderer.InterlaceFlags.HasFlag(InterlaceFlags.IsInterlaced))
                {
                    // Deinterlace using blend
                    result.Add(Deinterlace);
                }

                // Pre resize shaders, followed by NEDI image doubler
                result.Add(PreProcess);

                var size = Renderer.VideoSize;

                // Use NEDI once only.
                // Note: To use NEDI as many times as required to get the image past target size,
                //       Change the following *if* to *while*
                if (IsUpscalingFrom(size)) // See RenderScriptChain for other comparer methods
                {
                    result.Add(Nedi);
                    size = DoubleSize(size);
                }

                if (IsDownscalingFrom(size))
                {
                    // Use linear light for downscaling
                    result.Add(ToLinear);
                    result.Add(ResizeToTarget);
                    result.Add(ToGamma);
                }
                else
                {
                    // Otherwise, scale with gamma light
                    result.Add(ResizeToTarget);
                }

                if (!Is1080OrHigher(Renderer.VideoSize))
                {
                    // Sharpen only if video isn't full HD
                    result.Add(PostProcess);
                }

                return result.ToArray();
            }

            private static Size DoubleSize(Size size)
            {
                return new Size(size.Width*2, size.Height*2);
            }

            private static bool Is1080OrHigher(Size size)
            {
                return size.Height >= 1080;
            }
        }
    }
}
