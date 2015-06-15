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
            this.panel1 = new System.Windows.Forms.Panel();
            this.checkBoxDisable = new System.Windows.Forms.CheckBox();
            this.label1 = new System.Windows.Forms.Label();
            this.panel1.SuspendLayout();
            this.SuspendLayout();
            // 
            // downloadButton
            // 
            this.downloadButton.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.downloadButton.Location = new System.Drawing.Point(264, 242);
            this.downloadButton.Name = "downloadButton";
            this.downloadButton.Size = new System.Drawing.Size(75, 23);
            this.downloadButton.TabIndex = 0;
            this.downloadButton.Text = "&Download...";
            this.downloadButton.UseVisualStyleBackColor = true;
            this.downloadButton.Click += new System.EventHandler(this.downloadButton_Click);
            // 
            // forgetUpdate
            // 
            this.forgetUpdate.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.forgetUpdate.DialogResult = System.Windows.Forms.DialogResult.Cancel;
            this.forgetUpdate.Location = new System.Drawing.Point(345, 242);
            this.forgetUpdate.Name = "forgetUpdate";
            this.forgetUpdate.Size = new System.Drawing.Size(86, 23);
            this.forgetUpdate.TabIndex = 1;
            this.forgetUpdate.Text = "&Forget Update";
            this.forgetUpdate.UseVisualStyleBackColor = true;
            this.forgetUpdate.Click += new System.EventHandler(this.forgetUpdate_Click);
            // 
            // CloseButton
            // 
            this.CloseButton.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.CloseButton.DialogResult = System.Windows.Forms.DialogResult.Cancel;
            this.CloseButton.Location = new System.Drawing.Point(437, 242);
            this.CloseButton.Name = "CloseButton";
            this.CloseButton.Size = new System.Drawing.Size(75, 23);
            this.CloseButton.TabIndex = 2;
            this.CloseButton.Text = "&Cancel";
            this.CloseButton.UseVisualStyleBackColor = true;
            // 
            // changelogBox
            // 
            this.changelogBox.BackColor = System.Drawing.SystemColors.Control;
            this.changelogBox.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.changelogBox.Dock = System.Windows.Forms.DockStyle.Fill;
            this.changelogBox.Font = new System.Drawing.Font("Consolas", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.changelogBox.Location = new System.Drawing.Point(1, 1);
            this.changelogBox.Name = "changelogBox";
            this.changelogBox.ReadOnly = true;
            this.changelogBox.Size = new System.Drawing.Size(499, 167);
            this.changelogBox.TabIndex = 0;
            this.changelogBox.Text = "";
            this.changelogBox.WordWrap = false;
            // 
            // panel1
            // 
            this.panel1.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panel1.BackColor = System.Drawing.SystemColors.ControlDark;
            this.panel1.Controls.Add(this.changelogBox);
            this.panel1.Location = new System.Drawing.Point(12, 29);
            this.panel1.Name = "panel1";
            this.panel1.Padding = new System.Windows.Forms.Padding(1);
            this.panel1.Size = new System.Drawing.Size(501, 169);
            this.panel1.TabIndex = 3;
            // 
            // checkBoxDisable
            // 
            this.checkBoxDisable.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.checkBoxDisable.AutoSize = true;
            this.checkBoxDisable.Location = new System.Drawing.Point(13, 206);
            this.checkBoxDisable.Name = "checkBoxDisable";
            this.checkBoxDisable.Size = new System.Drawing.Size(430, 17);
            this.checkBoxDisable.TabIndex = 4;
            this.checkBoxDisable.Text = "Never check for updates (you can also change this in update checker\'s config dial" +
    "og)";
            this.checkBoxDisable.UseVisualStyleBackColor = true;
            this.checkBoxDisable.CheckedChanged += new System.EventHandler(this.checkBoxDisable_CheckedChanged);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(12, 9);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(52, 13);
            this.label1.TabIndex = 5;
            this.label1.Text = "Changes:";
            // 
            // UpdateCheckerNewVersionForm
            // 
            this.AcceptButton = this.downloadButton;
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.CancelButton = this.forgetUpdate;
            this.ClientSize = new System.Drawing.Size(525, 277);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.checkBoxDisable);
            this.Controls.Add(this.panel1);
            this.Controls.Add(this.CloseButton);
            this.Controls.Add(this.forgetUpdate);
            this.Controls.Add(this.downloadButton);
            this.MinimizeBox = false;
            this.Name = "UpdateCheckerNewVersionForm";
            this.ShowIcon = false;
            this.ShowInTaskbar = false;
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
            this.Text = "New Version Available";
            this.panel1.ResumeLayout(false);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button downloadButton;
        private System.Windows.Forms.Button forgetUpdate;
        private System.Windows.Forms.Button CloseButton;
        private System.Windows.Forms.RichTextBox changelogBox;
        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.CheckBox checkBoxDisable;
        private System.Windows.Forms.Label label1;
    }
}