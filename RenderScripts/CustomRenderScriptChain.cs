using System;
using Mpdn.RenderScript.Mpdn;

namespace Mpdn.RenderScript
{
    public abstract class CustomRenderScriptChain : RenderScriptChain
    {
        public override ScriptDescriptor Descriptor
        {
            get
            {
                return new ScriptDescriptor
                {
                    Name = "Custom Render Script Chain",
                    Description = "A customized render script chain (Advanced)",
                    Guid = new Guid("B0AD7BE7-A86D-4BE4-A750-4362FEF28A55")
                };
            }
        }
    }
}
