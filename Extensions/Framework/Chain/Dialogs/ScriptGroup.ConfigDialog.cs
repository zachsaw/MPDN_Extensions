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

namespace Mpdn.Extensions.Framework.Chain.Dialogs
{
    public partial class ScriptGroupDialog<T, TScript> : ScriptGroupDialogBase<T, TScript>
        where T : ITaggedProcess
        where TScript : class, IScript
    {
        public ScriptGroupDialog()
        {
            InitializeComponent();
        }

        protected override void LoadSettings()
        {
            m_ChainList.PresetList = Settings.Options;
            m_ChainList.SelectedIndex = Settings.SelectedIndex;
            HotkeyBox.Text = Settings.Hotkey;
        }

        protected override void SaveSettings()
        {
            Settings.Options = m_ChainList.PresetList;
            Settings.SelectedIndex = m_ChainList.SelectedIndex;
            Settings.Hotkey = HotkeyBox.Text;
        }
    }

    public class ScriptGroupDialogBase<T, TScript> : ScriptConfigDialog<ScriptGroup<T,TScript>> 
        where T : ITaggedProcess
        where TScript : class, IScript
    { }
}