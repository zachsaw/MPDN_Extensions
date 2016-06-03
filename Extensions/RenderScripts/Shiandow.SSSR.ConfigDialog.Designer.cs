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
    namespace Shiandow.SuperRes
    {
        partial class SSSRConfigDialog
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
            this.OverSharpSetter = new System.Windows.Forms.NumericUpDown();
            this.label1 = new System.Windows.Forms.Label();
            this.ButtonOK = new System.Windows.Forms.Button();
            this.ButtonCancel = new System.Windows.Forms.Button();
            this.label5 = new System.Windows.Forms.Label();
            this.PassesSetter = new System.Windows.Forms.NumericUpDown();
            this.PrescalerBox = new System.Windows.Forms.ComboBox();
            this.label6 = new System.Windows.Forms.Label();
            this.ConfigButton = new System.Windows.Forms.Button();
            this.LocalitySetter = new System.Windows.Forms.NumericUpDown();
            this.label8 = new System.Windows.Forms.Label();
            this.ModifyButton = new System.Windows.Forms.Button();
            this.ModeBox = new System.Windows.Forms.ComboBox();
            this.label2 = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.OverSharpSetter)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.PassesSetter)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.LocalitySetter)).BeginInit();
            this.SuspendLayout();
            // 
            // OverSharpSetter
            // 
            this.OverSharpSetter.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.OverSharpSetter.DecimalPlaces = 2;
            this.OverSharpSetter.Increment = new decimal(new int[] {
            1,
            0,
            0,
            65536});
            this.OverSharpSetter.Location = new System.Drawing.Point(223, 38);
            this.OverSharpSetter.Maximum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.OverSharpSetter.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            -2147483648});
            this.OverSharpSetter.Name = "OverSharpSetter";
            this.OverSharpSetter.Size = new System.Drawing.Size(44, 20);
            this.OverSharpSetter.TabIndex = 3;
            this.OverSharpSetter.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(12, 40);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(87, 13);
            this.label1.TabIndex = 2;
            this.label1.Text = "Over Sharpening";
            // 
            // ButtonOK
            // 
            this.ButtonOK.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.ButtonOK.Cursor = System.Windows.Forms.Cursors.Default;
            this.ButtonOK.DialogResult = System.Windows.Forms.DialogResult.OK;
            this.ButtonOK.Location = new System.Drawing.Point(111, 171);
            this.ButtonOK.Name = "ButtonOK";
            this.ButtonOK.Size = new System.Drawing.Size(75, 23);
            this.ButtonOK.TabIndex = 8;
            this.ButtonOK.Text = "OK";
            this.ButtonOK.UseVisualStyleBackColor = true;
            // 
            // ButtonCancel
            // 
            this.ButtonCancel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.ButtonCancel.Cursor = System.Windows.Forms.Cursors.Default;
            this.ButtonCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
            this.ButtonCancel.Location = new System.Drawing.Point(192, 171);
            this.ButtonCancel.Name = "ButtonCancel";
            this.ButtonCancel.Size = new System.Drawing.Size(75, 23);
            this.ButtonCancel.TabIndex = 9;
            this.ButtonCancel.Text = "Cancel";
            this.ButtonCancel.UseVisualStyleBackColor = true;
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(12, 14);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(41, 13);
            this.label5.TabIndex = 11;
            this.label5.Text = "Passes";
            // 
            // PassesSetter
            // 
            this.PassesSetter.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.PassesSetter.Location = new System.Drawing.Point(223, 12);
            this.PassesSetter.Maximum = new decimal(new int[] {
            10,
            0,
            0,
            0});
            this.PassesSetter.Name = "PassesSetter";
            this.PassesSetter.Size = new System.Drawing.Size(44, 20);
            this.PassesSetter.TabIndex = 2;
            this.PassesSetter.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.PassesSetter.Value = new decimal(new int[] {
            2,
            0,
            0,
            0});
            // 
            // PrescalerBox
            // 
            this.PrescalerBox.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.PrescalerBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.PrescalerBox.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.PrescalerBox.FormattingEnabled = true;
            this.PrescalerBox.Location = new System.Drawing.Point(69, 117);
            this.PrescalerBox.Name = "PrescalerBox";
            this.PrescalerBox.Size = new System.Drawing.Size(198, 21);
            this.PrescalerBox.TabIndex = 0;
            this.PrescalerBox.SelectedIndexChanged += new System.EventHandler(this.SelectionChanged);
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(12, 120);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(51, 13);
            this.label6.TabIndex = 13;
            this.label6.Text = "Prescaler";
            // 
            // ConfigButton
            // 
            this.ConfigButton.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.ConfigButton.Location = new System.Drawing.Point(15, 144);
            this.ConfigButton.Name = "ConfigButton";
            this.ConfigButton.Size = new System.Drawing.Size(171, 21);
            this.ConfigButton.TabIndex = 6;
            this.ConfigButton.Text = "Configure Prescaler";
            this.ConfigButton.UseVisualStyleBackColor = true;
            this.ConfigButton.Click += new System.EventHandler(this.ConfigButtonClick);
            // 
            // LocalitySetter
            // 
            this.LocalitySetter.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.LocalitySetter.DecimalPlaces = 2;
            this.LocalitySetter.Location = new System.Drawing.Point(223, 64);
            this.LocalitySetter.Maximum = new decimal(new int[] {
            100,
            0,
            0,
            65536});
            this.LocalitySetter.Minimum = new decimal(new int[] {
            10,
            0,
            0,
            65536});
            this.LocalitySetter.Name = "LocalitySetter";
            this.LocalitySetter.Size = new System.Drawing.Size(44, 20);
            this.LocalitySetter.TabIndex = 6;
            this.LocalitySetter.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.LocalitySetter.Value = new decimal(new int[] {
            10,
            0,
            0,
            65536});
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(12, 66);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(43, 13);
            this.label8.TabIndex = 16;
            this.label8.Text = "Locality";
            this.label8.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // ModifyButton
            // 
            this.ModifyButton.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.ModifyButton.Location = new System.Drawing.Point(192, 144);
            this.ModifyButton.Name = "ModifyButton";
            this.ModifyButton.Size = new System.Drawing.Size(75, 21);
            this.ModifyButton.TabIndex = 17;
            this.ModifyButton.Text = "Modify List";
            this.ModifyButton.UseVisualStyleBackColor = true;
            this.ModifyButton.Click += new System.EventHandler(this.ModifyButtonClick);
            // 
            // ModeBox
            // 
            this.ModeBox.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.ModeBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.ModeBox.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.ModeBox.FormattingEnabled = true;
            this.ModeBox.Location = new System.Drawing.Point(69, 90);
            this.ModeBox.Name = "ModeBox";
            this.ModeBox.Size = new System.Drawing.Size(198, 21);
            this.ModeBox.TabIndex = 18;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(12, 93);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(34, 13);
            this.label2.TabIndex = 19;
            this.label2.Text = "Mode";
            // 
            // SSSRConfigDialog
            // 
            this.AcceptButton = this.ButtonOK;
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.CancelButton = this.ButtonCancel;
            this.ClientSize = new System.Drawing.Size(279, 206);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.ModeBox);
            this.Controls.Add(this.ModifyButton);
            this.Controls.Add(this.label8);
            this.Controls.Add(this.LocalitySetter);
            this.Controls.Add(this.ConfigButton);
            this.Controls.Add(this.label6);
            this.Controls.Add(this.PrescalerBox);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.PassesSetter);
            this.Controls.Add(this.ButtonCancel);
            this.Controls.Add(this.ButtonOK);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.OverSharpSetter);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "SSSRConfigDialog";
            this.ShowIcon = false;
            this.ShowInTaskbar = false;
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
            this.Text = "SSSR Settings";
            ((System.ComponentModel.ISupportInitialize)(this.OverSharpSetter)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.PassesSetter)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.LocalitySetter)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

            }

            #endregion

            private System.Windows.Forms.NumericUpDown OverSharpSetter;
            private System.Windows.Forms.Label label1;
            private System.Windows.Forms.Button ButtonOK;
            private System.Windows.Forms.Button ButtonCancel;
            private System.Windows.Forms.Label label5;
            private System.Windows.Forms.NumericUpDown PassesSetter;
            private System.Windows.Forms.ComboBox PrescalerBox;
            private System.Windows.Forms.Label label6;
            private System.Windows.Forms.Button ConfigButton;
            private System.Windows.Forms.NumericUpDown LocalitySetter;
            private System.Windows.Forms.Label label8;
            private System.Windows.Forms.Button ModifyButton;
            private System.Windows.Forms.ComboBox ModeBox;
            private System.Windows.Forms.Label label2;
        }
    }
}
