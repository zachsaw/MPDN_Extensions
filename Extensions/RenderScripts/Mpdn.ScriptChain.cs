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

using System;
using System.Linq;
using Mpdn.Extensions.Framework.RenderChain;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Mpdn.ScriptChain
    {
        public class ScriptChain : PresetCollection
        {
            public override IFilter CreateFilter(IFilter input)
            {
                return Options.Aggregate(input, (result, chain) => result + chain);
            }
        }

        public class ScriptChainScript : RenderChainUi<ScriptChain, ScriptChainDialog>
        {
            protected override string ConfigFileName
            {
                get { return "Mpdn.ScriptChain"; }
            }

            public override string Category
            {
                get { return "Meta"; }
            }

            public override ExtensionUiDescriptor Descriptor
            {
                get
                {
                    return new ExtensionUiDescriptor
                    {
                        Guid = new Guid("3A462015-2D92-43AC-B559-396DACF896C3"),
                        Name = "Script Chain",
                        Description = Settings.Options.Count > 0
                            ? string.Join(" ➔ ", Settings.Options.Select(x => x.Name))
                            : "Chains together multiple renderscripts"
                    };
                }
            }
        }
    }
}