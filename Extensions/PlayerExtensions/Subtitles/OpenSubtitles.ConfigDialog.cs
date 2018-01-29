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

using Mpdn.Extensions.Framework.Config;

namespace Mpdn.Extensions.PlayerExtensions.Subtitles
{
    public partial class OpenSubtitlesConfigDialog : OpenSubtitlesConfigBase
    {
        public OpenSubtitlesConfigDialog()
        {
            InitializeComponent();
            cultureBindingSource.DataSource = OpenSubtitlesLanguageHandler.GetListCulture();
            comboBoxPrefLanguage.DataSource = cultureBindingSource;
            comboBoxPrefLanguage.DropDownClosed += OpenSubtitlesLanguageHandler.ComboBoxPrefLanguageDropDownClosed;
            comboBoxPrefLanguage.KeyDown += OpenSubtitlesLanguageHandler.ComboBoxPrefLanguageKeyDown;
            comboBoxPrefLanguage.KeyPress += OpenSubtitlesLanguageHandler.ComboBoxPrefLanguageKeyPress;
        }

        protected override void LoadSettings()
        {
            if (Settings.PreferedLanguage != null)
            {
                comboBoxPrefLanguage.SelectedValue = Settings.PreferedLanguage;
            }
            else
            {
                comboBoxPrefLanguage.SelectedItem = OpenSubtitlesLanguageHandler.InvariantCulture;
            }
        }

        protected override void SaveSettings()
        {
            if (comboBoxPrefLanguage.SelectedItem.Equals(OpenSubtitlesLanguageHandler.InvariantCulture))
            {
                Settings.PreferedLanguage = null;
            }
            else
            {
                Settings.PreferedLanguage = (string) comboBoxPrefLanguage.SelectedValue;
            }

        }
    }

    public class OpenSubtitlesConfigBase : ScriptConfigDialog<OpenSubtitlesSettings>
    {
    }
}
