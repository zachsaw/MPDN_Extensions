using System;
using Mpdn.RenderScript.Mpdn.ImageProcessor;
using Mpdn.RenderScript.Mpdn.Resizer;
using Mpdn.RenderScript.Shiandow.Chroma;
using Mpdn.RenderScript.Shiandow.Nedi;

namespace Mpdn.RenderScript
{
    namespace MyOwnUniqueNameSpace // e.g. Replace with your user name
    {
        public class MyRenderChain : CombinedChain
        {
            private readonly string[] Deinterlace = {@"MPC-HC\Deinterlace (blend).hlsl"};
            private readonly string[] PostProcess = {@"SweetFX\LumaSharpen.hlsl"};
            private readonly string[] PreProcess = {@"SweetFX\Bloom.hlsl", @"SweetFX\LiftGammaGain.hlsl"};
            private readonly string[] ToGamma = {@"ConvertToGammaLight.hlsl"};
            private readonly string[] ToLinear = {@"ConvertToLinearLight.hlsl"};

            protected override void BuildChain(FilterChain chain)
            {
                // Scale chroma first (this bypasses MPDN's chroma scaler)
                chain.Add(new BicubicChroma {Preset = Presets.MitchellNetravali});

                if (Renderer.InterlaceFlags.HasFlag(InterlaceFlags.IsInterlaced))
                {
                    // Deinterlace using blend
                    chain.Add(new ImageProcessor {ShaderFileNames = Deinterlace});
                }

                // Pre resize shaders, followed by NEDI image doubler
                chain.Add(new ImageProcessor {ShaderFileNames = PreProcess});

                // Use NEDI once only.
                // Note: To use NEDI as many times as required to get the image past target size,
                //       Change the following *if* to *while*
                if (IsUpscalingFrom(chain)) // See CombinedChain for other comparer methods
                {
                    chain.Add(new Nedi {AlwaysDoubleImage = true});
                }

                if (IsDownscalingFrom(chain))
                {
                    // Use linear light for downscaling
                    chain.Add(new ImageProcessor {ShaderFileNames = ToLinear});
                    chain.Add(new Resizer {ResizerOption = ResizerOption.TargetSize100Percent});
                    chain.Add(new ImageProcessor {ShaderFileNames = ToGamma});
                }
                else
                {
                    // Otherwise, scale with gamma light
                    chain.Add(new Resizer {ResizerOption = ResizerOption.TargetSize100Percent});
                }

                if (Renderer.VideoSize.Width >= 1920)
                {
                    // Sharpen only if video isn't full HD
                    chain.Add(new ImageProcessor {ShaderFileNames = PostProcess});
                }
            }
        }

        public class MyRenderScript : RenderChainUi<MyRenderChain>
        {
            protected override RenderScriptDescriptor ScriptDescriptor
            {
                get
                {
                    return new RenderScriptDescriptor
                    {
                        Name = "Custom Render Script Chain",
                        Description = "A customized render script chain (Advanced)",
                        Guid = new Guid("B0AD7BE7-A86D-4BE4-A750-4362FEF28A55"),
                        Copyright = "" // Optional field
                    };
                }
            }
        }
    }
}