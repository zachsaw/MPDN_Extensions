using System;
using Mpdn.RenderScript;
using Mpdn.RenderScript.Mpdn.Resizer;

namespace Mpdn.RenderScripts
{
    namespace Example
    {
        public class DirectCompute : RenderChain
        {
            protected override string ShaderPath
            {
                get { return "Examples"; }
            }

            public override IFilter CreateFilter(IResizeableFilter sourceFilter)
            {
                if (!Renderer.IsDx11Avail)
                    return new NullFilter(); // display blank screen on purpose

                // get MPDN to scale image to target size first
                sourceFilter += new Resizer { ResizerOption = ResizerOption.TargetSize100Percent };

                // apply our blue tint
                var blueTint = CompileShader11("BlueTintDirectCompute.hlsl", "cs_5_0");
                var width = sourceFilter.OutputSize.Width;
                var height = sourceFilter.OutputSize.Height;
                return new DirectComputeFilter(blueTint, width/32 + 1, height/32 + 1, 1, new[] {0.25f, 0.5f, 0.75f},
                    sourceFilter);
            }
        }

        public class DirectComputeExample : RenderChainUi<DirectCompute>
        {
            public override ExtensionUiDescriptor Descriptor
            {
                get
                {
                    return new ExtensionUiDescriptor
                    {
                        Name = "DirectCompute Blue Tint Example",
                        Description = "(Example) Applies a blue tint over the image using DirectCompute",
                        Guid = new Guid("2BAD9125-6474-42D4-9C65-9A03DE3280AF"),
                        Copyright = "" // Optional field
                    };
                }
            }
        }
    }
}
