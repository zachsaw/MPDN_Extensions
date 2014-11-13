using System.Collections.Generic;
using System.Drawing;
using Mpdn.RenderScript.Mpdn.ImageProcessor;
using Mpdn.RenderScript.Mpdn.Resizer;
using Mpdn.RenderScript.Shiandow.Nedi;

namespace Mpdn.RenderScript
{
    namespace MyOwnUniqueNameSpace // e.g. Replace with your user name
    {
        public class MyRenderScript : CustomRenderScriptChain
        {
            private readonly string[] PreResizeShaderfiles = { @"SweetFX\Bloom.hlsl", /* add more files here (separate with comma) ... */ };
            private readonly string[] PostResizeShaderfiles = { @"SweetFX\LumaSharpen.hlsl", /* add more files here (separate with comma) ... */ };

            private RenderScript Nedi, PreProcess, PostProcess, ToLinear, ToGamma, ResizeToTarget;

            protected override RenderScript[] CreateScripts()
            {
                return new[]
                {
                    Nedi = NediScaler.Create(forced: true),
                    PreProcess = ImageProcessor.Create(PreResizeShaderfiles),
                    PostProcess = ImageProcessor.Create(PostResizeShaderfiles),
                    ToLinear = ImageProcessor.Create(new[] {@"ConvertToLinearLight.hlsl"}),
                    ToGamma = ImageProcessor.Create(new[] {@"ConvertToGammaLight.hlsl"}),
                    ResizeToTarget = Resizer.Create(option: ResizerOption.TargetSize100Percent),
                    // Add more scripts here ...
                };
            }

            protected override RenderScript[] GetScriptChain()
            {
                var result = new List<RenderScript>();

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
                    // For final downscaling, use linear light scaling
                    result.Add(ToLinear);
                    result.Add(ResizeToTarget);
                    result.Add(ToGamma);
                }
                else
                {
                    // Otherwise, use scale with gamma light
                    result.Add(ResizeToTarget);
                }

                // Post resize shaders
                result.Add(PostProcess);

                return result.ToArray();
            }

            private static Size DoubleSize(Size size)
            {
                return new Size(size.Width*2, size.Height*2);
            }
        }
    }
}
