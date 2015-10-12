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

using Mpdn.Extensions.Framework.Config;
using Mpdn.Extensions.Framework.Controls;
using Mpdn.Extensions.Framework.AudioChain;
using Mpdn.AudioScript;

namespace Mpdn.Extensions.AudioScripts
{
    namespace Mpdn.ScriptChain
    {
        public partial class ScriptChainDialog : ScriptChainDialogBase
        {
            public ScriptChainDialog()
            {
                InitializeComponent();
            }

            protected override void LoadSettings()
            {
                m_ChainList.PresetList = Settings.Options;
            }

            protected override void SaveSettings()
            {
                Settings.Options = m_ChainList.PresetList;
            }
        }

        public class AudioScriptChainList : ChainList<Audio, IAudioScript>
        {
        }

        public class ScriptChainDialogBase : ScriptConfigDialog<ScriptChain>
        {
        }
    }
}
