using System.Collections.Generic;
using System.Drawing;
using Mpdn.RenderScript.Mpdn.ImageProcessor;
using Mpdn.RenderScript.Mpdn.Resizer;
using Mpdn.RenderScript.Scaler;
using Mpdn.RenderScript.Shiandow.Chroma;
using Mpdn.RenderScript.Shiandow.Nedi;

namespace Mpdn.RenderScript
{
    namespace MyOwnUniqueNameSpace // e.g. Replace with your user name
    {
        public class MyRenderScript : CustomRenderScriptChain
        {
            private RenderScript ScaleChroma, Nedi, PreProcess, PostProcess, Deinterlace, ToLinear, ToGamma, ResizeToTarget;

            protected override IList<RenderScript> CreateScripts()
            {
                // Declare all the scripts we will be using
                return new[]
                {
                    ScaleChroma    = ChromaScaler.Create(preset: Presets.Spline),
                    Nedi           = NediScaler.Create(forced: true),
                    PreProcess     = ImageProcessor.Create(@"SweetFX\Bloom.hlsl", @"SweetFX\LiftGammaGain.hlsl"),
                    PostProcess    = ImageProcessor.Create(@"SweetFX\LumaSharpen.hlsl"),
                    Deinterlace    = ImageProcessor.Create(@"MPC-HC\Deinterlace (blend).hlsl"),
                    ToLinear       = ImageProcessor.Create(@"ConvertToLinearLight.hlsl"),
                    ToGamma        = ImageProcessor.Create(@"ConvertToGammaLight.hlsl"),
                    // Note: By specifying upscaler and downscaler in Resizer, we bypass MPDN's scaler settings altogether
                    ResizeToTarget = Resizer.Create(option: ResizerOption.TargetSize100Percent, upscaler: new Softcubic(1.0f), downscaler: new Softcubic(1.0f)),
                    // Add more scripts here ...
                };
            }

            protected override IList<RenderScript> GetScriptChain()
            {
                var chain = new List<RenderScript>();

                // Scale chroma first (this bypasses MPDN's chroma scaler)
                chain.Add(ScaleChroma);

                if (Renderer.InterlaceFlags.HasFlag(InterlaceFlags.IsInterlaced))
                {
                    // Deinterlace using blend
                    chain.Add(Deinterlace);
                }

                // Pre resize shaders, followed by NEDI image doubler
                chain.Add(PreProcess);

                // Use NEDI once only.
                // Note: To use NEDI as many times as required to get the image past target size,
                //       Change the following *if* to *while*
                if (IsUpscalingFrom(chain)) // See RenderScriptChain for other comparer methods
                {
                    chain.Add(Nedi);
                }

                if (IsDownscalingFrom(chain))
                {
                    // Use linear light for downscaling
                    chain.Add(ToLinear);
                    chain.Add(ResizeToTarget);
                    chain.Add(ToGamma);
                }
                else
                {
                    // Otherwise, scale with gamma light
                    chain.Add(ResizeToTarget);
                }

                if (!Is1080OrHigher(Renderer.VideoSize))
                {
                    // Sharpen only if video isn't full HD
                    chain.Add(PostProcess);
                }

                return chain;
            }

            private static bool Is1080OrHigher(Size size)
            {
                return size.Height >= 1080;
            }
        }
    }
}
