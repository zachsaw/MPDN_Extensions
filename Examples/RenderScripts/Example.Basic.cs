using System;
using Mpdn.RenderScript;
using Mpdn.RenderScript.Mpdn.Resizer;

namespace Mpdn.RenderScripts
{
    namespace Example
    {
        public class Basic : RenderChain
        {
            protected override string ShaderPath
            {
                get { return "Examples"; }
            }

            public override IFilter CreateFilter(IResizeableFilter sourceFilter)
            {
                // get MPDN to scale image to target size first
                sourceFilter += new Resizer { ResizerOption = ResizerOption.TargetSize100Percent };

                // apply our blue tint
                var blueTint = CompileShader("BlueTintSm3.hlsl");
                return new ShaderFilter(blueTint, false, new[] {0.25f, 0.5f, 0.75f}, sourceFilter);
            }
        }

        public class BasicExample : RenderChainUi<Basic>
        {
            public override ExtensionUiDescriptor Descriptor
            {
                get
                {
                    return new ExtensionUiDescriptor
                    {
                        Name = "SM3.0 Blue Tint Example",
                        Description = "(Example) Applies a blue tint over the image using Shader Model 3.0",
                        Guid = new Guid("3682DAD5-067C-4537-B540-BE86A7C3527A"),
                        Copyright = "" // Optional field
                    };
                }
            }
        }
    }
}
