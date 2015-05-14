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
using System.IO;
using System.Windows.Forms;
using Mpdn.Config;

namespace Mpdn.RenderScript
{
    namespace Mpdn.ScriptedRenderChain
    {
        public partial class ScriptedRenderChainConfigDialog : ScriptedRenderChainConfigDialogBase
        {
            public ScriptedRenderChainConfigDialog()
            {
                InitializeComponent();

                Icon = PlayerControl.ApplicationIcon;
            }

            protected override void LoadSettings()
            {
                var file = Settings.ScriptFileName;
                if (File.Exists(file))
                {
                    textBoxScript.Text = File.ReadAllText(file);
                }
                else
                {
                    if (!string.IsNullOrWhiteSpace(file))
                    {
                        MessageBox.Show(this, string.Format("Script file '{0}' not found. Default loaded.", file), "Error");
                    }
                    textBoxScript.Text = Helpers.DefaultScript;
                }

                textBoxScript.SelectionStart = 0;
                textBoxScript.SelectionLength = 0;
            }

            protected override void SaveSettings()
            {
                File.WriteAllText(Settings.ScriptFileName, textBoxScript.Text);
            }
        }

        public class ScriptedRenderChainConfigDialogBase : ScriptConfigDialog<ScriptedRenderChain>
        {
        }
    }
}
