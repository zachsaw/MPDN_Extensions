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
// 
using System;
using IBaseFilter = Mpdn.RenderScript.IFilter<Mpdn.IBaseTexture>;

namespace Mpdn.RenderScript
{
    namespace Example
    {
        public sealed class FrameRateHalver : RenderChain
        {
            private class FramerateHalvingFilter : BasicFilter
            {
                private int m_Counter;

                public FramerateHalvingFilter(IFilter inputFilter)
                    : base(inputFilter)
                {
                }

                protected override void Render(ITexture texture)
                {
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

            public override IFilter CreateFilter(IFilter sourceFilter)
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

            public override string Category
            {
                get { return "Example"; }
            }
        }
    }
}
