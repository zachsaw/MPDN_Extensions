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

namespace Mpdn.Extensions.Framework.Controls
{
        public partial class ScriptedRenderChainScriptEditorDialog : Form
        {
            private bool m_Changed;
            private string m_File;

            public ScriptedRenderChainScriptEditorDialog()
            {
                InitializeComponent();

                Icon = PlayerControl.ApplicationIcon;
                textBoxScript.Editor.TextChanged += (s, e) =>
                {
                    m_Changed = true;
                    Text = Path.GetFileName(m_File) + "*";
                };
            }

            public void LoadFile(string file)
            {
                m_File = file;
                textBoxScript.Editor.Text = File.ReadAllText(file);
                Text = Path.GetFileName(m_File);
                m_Changed = false;
            }

            public void SaveFile()
            {
                File.WriteAllText(m_File, textBoxScript.Editor.Text);
                m_Changed = false;
                Text = Path.GetFileName(m_File);
            }

            private void CloseClick(object sender, System.EventArgs e)
            {
                if (m_Changed && !PromptSave()) 
                    return;

                Close();
            }

            private bool PromptSave()
            {
                var result = MessageBox.Show(this, "Do you want to save the changes?", "Save changes",
                    MessageBoxButtons.YesNoCancel, MessageBoxIcon.Question);
                if (result == DialogResult.Cancel)
                    return false;

                if (result == DialogResult.Yes)
                {
                    SaveFile();
                }
                return true;
            }

            private void SaveClick(object sender, System.EventArgs e)
            {
                if (!m_Changed)
                    return;

                SaveFile();
            }

            private void AboutClick(object sender, System.EventArgs e)
            {
                MessageBox.Show(this,
                    "This script editor uses AvalonEdit (MIT License).\r\nCopyright (c) 2014 AlphaSierraPapa for the SharpDevelopTeam.",
                    "About", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }

            private void FormShown(object sender, System.EventArgs e)
            {
                HourGlass.Enabled = false;
            }
        }
}
