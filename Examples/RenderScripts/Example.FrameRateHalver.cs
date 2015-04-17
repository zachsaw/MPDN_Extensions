using System;
using System.Collections.Generic;
using System.Linq;
using Mpdn.RenderScript;
using IBaseFilter = Mpdn.RenderScript.IFilter<Mpdn.IBaseTexture>;

namespace Mpdn.RenderScripts
{
    namespace Example
    {
        public class FrameRateHalver : RenderChain
        {
            private class FramerateHalvingFilter : Filter
            {
                private int m_Counter;

                public FramerateHalvingFilter(IBaseFilter inputFilter)
                    : base(inputFilter)
                {
                }

                protected override void Render(IList<IBaseTexture> inputs)
                {
                    var texture = inputs.OfType<ITexture>().SingleOrDefault();
                    if (texture == null)
                        return;

                    // Render all frames but only present half of them
                    // In real life scenario, you'd probably want to use Renderer.RenderQueue[i].Frame
                    // to do something useful with the current frame.
                    // (Note: Renderer.RenderQueue.First().Frame is the frame before the current, while 
                    //        Renderer.RenderQueue.Last().Frame is earliest frame in the queue.
                    //        Renderer.RenderQueue will have no elements to start off with!)
                    Renderer.Render(OutputTexture, texture, false);

                    // Note: To get actual odd/even frame number, you should calculate from 
                    //       Renderer.FrameRateHz and Renderer.FrameTimeStampMicrosec instead of relying on m_Counter
                    // Note: Setting PresentFrame to false will cause MPDN to disable FluidMotion
                    Renderer.PresentFrame = ((m_Counter % 2) == 0) || Renderer.RenderQueue.Count <= 1;
                    m_Counter++;
                }
            }

            protected override string ShaderPath
            {
                get { return "Examples"; }
            }

            public override IFilter CreateFilter(IResizeableFilter sourceFilter)
            {
                // apply the halving filter
                return new FramerateHalvingFilter(sourceFilter);
            }
        }

        public class FrameRateHalverExample : RenderChainUi<FrameRateHalver>
        {
            public override ExtensionUiDescriptor Descriptor
            {
                get
                {
                    return new ExtensionUiDescriptor
                    {
                        Name = "Framerate Halving Example",
                        Description = "(Example) Halves the framerate of the source (render all frames but only present one in two)",
                        Guid = new Guid("2541F581-6419-4F9C-8D4F-87FA23318FB6"),
                        Copyright = "" // Optional field
                    };
                }
            }
        }
    }
}
