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
using System.Collections.Generic;
using System.Linq;

namespace Mpdn.RenderScript
{
    namespace Mpdn.ScriptChain
    {
        public class ScriptChain : RenderChain
        {
            public List<IRenderChainUi> ScriptList { get; set; }

            public ScriptChain()
            {
                ScriptList = new List<IRenderChainUi>();
            }

            public override IFilter CreateFilter(IResizeableFilter sourceFilter)
            {
                return ScriptList.Select(pair => pair.GetChain()).Aggregate(sourceFilter, (a, b) => a + b);
            }
        }

        public class ScriptChainScript : RenderChainUi<ScriptChain, ScriptChainDialog>
        {
            protected override string ConfigFileName
            {
                get { return "Mpdn.ScriptChain"; }
            }

            public override ExtensionUiDescriptor Descriptor
            {
                get
                {
                    return new ExtensionUiDescriptor
                    {
                        Guid = new Guid("3A462015-2D92-43AC-B559-396DACF896C3"),
                        Name = "Script Chain",
                        Description = GetDescription(),
                    };
                }
            }

            private string GetDescription()
            {
                return Chain.ScriptList.Count == 0
                    ? "Chain of render scripts"
                    : string.Join(" ➔ ", Chain.ScriptList.Select(x => x.Descriptor.Name));
            }
        }
    }
}
