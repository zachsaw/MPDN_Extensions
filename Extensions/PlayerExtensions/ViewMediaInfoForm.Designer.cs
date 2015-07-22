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
namespace Mpdn.Extensions.PlayerExtensions
{
    partial class ViewMediaInfoForm
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
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.wb_info = new System.Windows.Forms.WebBrowser();
            this.SuspendLayout();
            // 
            // wb_info
            // 
            this.wb_info.Dock = System.Windows.Forms.DockStyle.Fill;
            this.wb_info.IsWebBrowserContextMenuEnabled = false;
            this.wb_info.Location = new System.Drawing.Point(0, 0);
            this.wb_info.MinimumSize = new System.Drawing.Size(20, 20);
            this.wb_info.Name = "wb_info";
            this.wb_info.Size = new System.Drawing.Size(724, 661);
            this.wb_info.TabIndex = 4;
            this.wb_info.WebBrowserShortcutsEnabled = false;
            // 
            // ViewMediaInfoForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(724, 661);
            this.Controls.Add(this.wb_info);
            this.KeyPreview = true;
            this.MinimizeBox = false;
            this.Name = "ViewMediaInfoForm";
            this.ShowInTaskbar = false;
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
            this.Text = "Media Info";
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.WebBrowser wb_info;
    }
}
