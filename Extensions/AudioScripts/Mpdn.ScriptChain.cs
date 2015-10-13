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
using System.Collections.Generic;
using System.Linq;
using System.Windows.Forms;
using Mpdn.Extensions.Framework.Chain;
using Mpdn.Extensions.Framework.AudioChain;
using Mpdn.AudioScript;

namespace Mpdn.Extensions.AudioScripts
{
    namespace Mpdn.ScriptChain
    {
        public class ScriptChain : PresetCollection<Audio, IAudioScript>
        {
            public override Audio Process(Audio input)
            {
                return ActiveOptions.Aggregate(input, (result, chain) => result + chain);
            }
        }

        public class ScriptChainScript : AudioChainUi<ScriptChain, ScriptChainDialog>
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
                        Guid = new Guid("34804677-65D8-4C1E-8BFE-B897749D617B"),
                        Name = "Script Chain",
                        Description = Settings.Options.Count > 0
                            ? string.Join(" ➔ ", Settings.Options.Select(x => x.Name))
                            : "Chains together multiple renderscripts"
                    };
                }
            }

            public override bool ShowConfigDialog(IWin32Window owner)
            {
                var result = base.ShowConfigDialog(owner);
                if (!result)
                    return false;

                // Activate new options
                Settings.Reset();
                Settings.Initialize();
                return true;
            }
        }
    }
}