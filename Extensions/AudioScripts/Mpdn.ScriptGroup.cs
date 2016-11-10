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
using Mpdn.Extensions.Framework.AudioChain;

namespace Mpdn.Extensions.AudioScripts
{
    namespace Mpdn.ScriptGroup
    {
        public class ScriptGroupScript : GeneralAudioChainUI<AudioScriptGroup, AudioScriptGroupDialog>
        {
            protected override string ConfigFileName
            {
                get { return "Mpdn.ScriptGroup"; }
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
                        Guid = new Guid("B89C92BD-30C4-42CA-B07C-6D6BF4ED7131"),
                        Name = "Script Group",
                        Description = Description()
                    };
                }
            }

            public string Description()
            {
                return (Settings.Options.Count > 0)
                    ? string.Join(", ",
                        Settings.Options.Select(x =>
                            (x == Settings.SelectedOption)
                                ? "[" + x.Name + "]"
                                : x.Name))
                    : "Picks one out of several audioscripts";
            }
        }
    }
}