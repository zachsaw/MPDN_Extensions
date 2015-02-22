using System;
using Mpdn.RenderScript;
using Mpdn.RenderScript.Mpdn.Resizer;

namespace Mpdn.RenderScripts
{
    namespace Example
    {
        public class Sm5 : RenderChain
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
                var blueTint = CompileShader11("BlueTintSm5.hlsl", "ps_5_0");
                return new Shader11Filter(blueTint, false, new[] {0.25f, 0.5f, 0.75f}, sourceFilter);
            }
        }

        public class Sm5Example : RenderChainUi<Sm5>
        {
            public override ExtensionUiDescriptor Descriptor
            {
                get
                {
                    return new ExtensionUiDescriptor
                    {
                        Name = "SM5.0 Blue Tint Example",
                        Description = "(Example) Applies a blue tint over the image using Shader Model 5.0",
                        Guid = new Guid("72594680-274B-464B-9BD0-6D99173B4555"),
                        Copyright = "" // Optional field
                    };
                }
            }
        }
    }
}
