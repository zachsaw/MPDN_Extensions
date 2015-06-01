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
using Mpdn.Extensions.Framework.Config;
using Mpdn.Extensions.Framework.Controls;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Mpdn.ScriptedRenderChain
    {
        public partial class ScriptedRenderChainConfigDialog : ScriptedRenderChainConfigDialogBase
        {
            public ScriptedRenderChainConfigDialog()
            {
                InitializeComponent();

                openFileDialog.Filter = "RenderScript files|*.rs|All files|*.*";
            }

            protected override void LoadSettings()
            {
                textBoxScript.Text = Settings.ScriptFileName;
            }

            protected override void SaveSettings()
            {
                Settings.ScriptFileName = textBoxScript.Text;
            }

            private void EditClick(object sender, System.EventArgs e)
            {
                var file = textBoxScript.Text;
                if (!EnsureFileExists(file)) 
                    return;

                using (new HourGlass())
                {
                    var editor = new ScriptedRenderChainScriptEditorDialog();
                    editor.LoadFile(file);
                    editor.ShowDialog(this);
                }
            }

            private bool EnsureFileExists(string file)
            {
                if (!File.Exists(file))
                {
                    if (MessageBox.Show(this, string.Format("Script file '{0}' not found.\r\nCreate a new file?", file),
                        "Confirm", MessageBoxButtons.YesNo, MessageBoxIcon.Question) != DialogResult.Yes)
                        return false;

                    File.WriteAllText(file, Helpers.DefaultScript);
                }
                return true;
            }

            private void SelectFileClick(object sender, System.EventArgs e)
            {
                openFileDialog.FileName = textBoxScript.Text;
                openFileDialog.InitialDirectory = MpdnPath.GetDirectoryName(openFileDialog.FileName);
                if (openFileDialog.ShowDialog(this) != DialogResult.OK)
                    return;

                textBoxScript.Text = openFileDialog.FileName;
                EnsureFileExists(textBoxScript.Text);
            }
        }

        public class ScriptedRenderChainConfigDialogBase : ScriptConfigDialog<ScriptedRenderChain>
        {
        }
    }
}
