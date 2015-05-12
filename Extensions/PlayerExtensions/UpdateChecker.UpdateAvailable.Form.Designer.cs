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
namespace Mpdn.PlayerExtensions.GitHub
{
    partial class UpdateCheckerNewVersionForm
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
            this.downloadButton = new System.Windows.Forms.Button();
            this.forgetUpdate = new System.Windows.Forms.Button();
            this.CloseButton = new System.Windows.Forms.Button();
            this.changelogBox = new System.Windows.Forms.RichTextBox();
            this.SuspendLayout();
            // 
            // downloadButton
            // 
            this.downloadButton.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.downloadButton.Location = new System.Drawing.Point(337, 133);
            this.downloadButton.Name = "downloadButton";
            this.downloadButton.Size = new System.Drawing.Size(75, 23);
            this.downloadButton.TabIndex = 0;
            this.downloadButton.Text = "Download";
            this.downloadButton.UseVisualStyleBackColor = true;
            this.downloadButton.Click += new System.EventHandler(this.downloadButton_Click);
            // 
            // forgetUpdate
            // 
            this.forgetUpdate.Anchor = System.Windows.Forms.AnchorStyles.Bottom;
            this.forgetUpdate.Location = new System.Drawing.Point(169, 133);
            this.forgetUpdate.Name = "forgetUpdate";
            this.forgetUpdate.Size = new System.Drawing.Size(86, 23);
            this.forgetUpdate.TabIndex = 1;
            this.forgetUpdate.Text = "Forget Update";
            this.forgetUpdate.UseVisualStyleBackColor = true;
            this.forgetUpdate.Click += new System.EventHandler(this.forgetUpdate_Click);
            // 
            // CloseButton
            // 
            this.CloseButton.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.CloseButton.Location = new System.Drawing.Point(12, 133);
            this.CloseButton.Name = "CloseButton";
            this.CloseButton.Size = new System.Drawing.Size(75, 23);
            this.CloseButton.TabIndex = 2;
            this.CloseButton.Text = "Close";
            this.CloseButton.UseVisualStyleBackColor = true;
            // 
            // changelogBox
            // 
            this.changelogBox.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.changelogBox.Location = new System.Drawing.Point(12, 12);
            this.changelogBox.Name = "changelogBox";
            this.changelogBox.ReadOnly = true;
            this.changelogBox.Size = new System.Drawing.Size(400, 115);
            this.changelogBox.TabIndex = 4;
            this.changelogBox.Text = "";
            // 
            // UpdateCheckerNewVersionForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(424, 168);
            this.Controls.Add(this.changelogBox);
            this.Controls.Add(this.CloseButton);
            this.Controls.Add(this.forgetUpdate);
            this.Controls.Add(this.downloadButton);
            this.Name = "UpdateCheckerNewVersionForm";
            this.ShowIcon = false;
            this.ShowInTaskbar = false;
            this.Text = "New Version Available";
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Button downloadButton;
        private System.Windows.Forms.Button forgetUpdate;
        private System.Windows.Forms.Button CloseButton;
        private System.Windows.Forms.RichTextBox changelogBox;
    }
}