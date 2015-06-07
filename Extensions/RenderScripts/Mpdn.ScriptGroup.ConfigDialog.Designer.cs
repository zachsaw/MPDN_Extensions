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
namespace Mpdn.Extensions.RenderScripts
{
    namespace Mpdn.ScriptGroup
    {
        partial class PresetGroupDialog
        {
            /// <summary> 
            /// Required designer variable.
            /// </summary>
            private System.ComponentModel.IContainer components = null;

            /// <summary> 
            /// Clean up any resources being used.
            /// </summary>
            /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
            protected override void Dispose(bool disposing)
            {
                if (disposing && (components != null))
                {
                    components.Dispose();
                }
            }

            #region Component Designer generated code

            /// <summary> 
            /// Required method for Designer support - do not modify 
            /// the contents of this method with the code editor.
            /// </summary>
            private void InitializeComponent()
            {
            this.buttonCancel = new System.Windows.Forms.Button();
            this.buttonOk = new System.Windows.Forms.Button();
            this.HotkeyLabel = new System.Windows.Forms.Label();
            this.HotkeyBox = new Framework.Controls.HotkeyBox();
            this.RenderChainList = new Framework.Controls.RenderChainList();
            this.SuspendLayout();
            // 
            // buttonCancel
            // 
            this.buttonCancel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.buttonCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
            this.buttonCancel.Location = new System.Drawing.Point(791, 521);
            this.buttonCancel.Name = "buttonCancel";
            this.buttonCancel.Size = new System.Drawing.Size(75, 23);
            this.buttonCancel.TabIndex = 1;
            this.buttonCancel.Text = "Cancel";
            this.buttonCancel.UseVisualStyleBackColor = true;
            // 
            // buttonOk
            // 
            this.buttonOk.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.buttonOk.DialogResult = System.Windows.Forms.DialogResult.OK;
            this.buttonOk.Location = new System.Drawing.Point(710, 521);
            this.buttonOk.Name = "buttonOk";
            this.buttonOk.Size = new System.Drawing.Size(75, 23);
            this.buttonOk.TabIndex = 0;
            this.buttonOk.Text = "OK";
            this.buttonOk.UseVisualStyleBackColor = true;
            // 
            // HotkeyLabel
            // 
            this.HotkeyLabel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.HotkeyLabel.AutoSize = true;
            this.HotkeyLabel.Location = new System.Drawing.Point(18, 526);
            this.HotkeyLabel.Name = "HotkeyLabel";
            this.HotkeyLabel.Size = new System.Drawing.Size(41, 13);
            this.HotkeyLabel.TabIndex = 11;
            this.HotkeyLabel.Text = "Hotkey";
            // 
            // HotkeyBox
            // 
            this.HotkeyBox.AcceptsReturn = true;
            this.HotkeyBox.AcceptsTab = true;
            this.HotkeyBox.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.HotkeyBox.BackColor = System.Drawing.SystemColors.Window;
            this.HotkeyBox.Location = new System.Drawing.Point(65, 523);
            this.HotkeyBox.Name = "HotkeyBox";
            this.HotkeyBox.ReadOnly = true;
            this.HotkeyBox.Size = new System.Drawing.Size(145, 20);
            this.HotkeyBox.TabIndex = 5;
            // 
            // RenderChainList
            // 
            this.RenderChainList.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.RenderChainList.Location = new System.Drawing.Point(18, 18);
            this.RenderChainList.Name = "RenderChainList";
            this.RenderChainList.SelectedIndex = -1;
            this.RenderChainList.Size = new System.Drawing.Size(848, 497);
            this.RenderChainList.TabIndex = 16;
            // 
            // PresetGroupDialog
            // 
            this.AcceptButton = this.buttonOk;
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.CancelButton = this.buttonCancel;
            this.ClientSize = new System.Drawing.Size(884, 562);
            this.Controls.Add(this.RenderChainList);
            this.Controls.Add(this.HotkeyBox);
            this.Controls.Add(this.HotkeyLabel);
            this.Controls.Add(this.buttonCancel);
            this.Controls.Add(this.buttonOk);
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.MinimumSize = new System.Drawing.Size(289, 172);
            this.Name = "PresetGroupDialog";
            this.Padding = new System.Windows.Forms.Padding(15);
            this.ShowIcon = false;
            this.ShowInTaskbar = false;
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
            this.Text = "Script Group";
            this.ResumeLayout(false);
            this.PerformLayout();

            }

            #endregion

            private System.Windows.Forms.Button buttonCancel;
            private System.Windows.Forms.Button buttonOk;
            private System.Windows.Forms.Label HotkeyLabel;
            private Framework.Controls.HotkeyBox HotkeyBox;
            private Framework.Controls.RenderChainList RenderChainList;
        }
    }
}  
