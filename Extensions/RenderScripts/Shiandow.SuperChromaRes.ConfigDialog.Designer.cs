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
        partial class SuperChromaResConfigDialog
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
            this.StrengthSetter = new System.Windows.Forms.NumericUpDown();
            this.label1 = new System.Windows.Forms.Label();
            this.ButtonOK = new System.Windows.Forms.Button();
            this.ButtonCancel = new System.Windows.Forms.Button();
            this.label5 = new System.Windows.Forms.Label();
            this.PassesSetter = new System.Windows.Forms.NumericUpDown();
            this.SoftnessSetter = new System.Windows.Forms.NumericUpDown();
            this.label6 = new System.Windows.Forms.Label();
            this.PrescalerBox = new System.Windows.Forms.CheckBox();
            ((System.ComponentModel.ISupportInitialize)(this.StrengthSetter)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.PassesSetter)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.SoftnessSetter)).BeginInit();
            this.SuspendLayout();
            // 
            // StrengthSetter
            // 
            this.StrengthSetter.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.StrengthSetter.DecimalPlaces = 2;
            this.StrengthSetter.Increment = new decimal(new int[] {
            1,
            0,
            0,
            65536});
            this.StrengthSetter.Location = new System.Drawing.Point(123, 38);
            this.StrengthSetter.Maximum = new decimal(new int[] {
            2,
            0,
            0,
            0});
            this.StrengthSetter.Name = "StrengthSetter";
            this.StrengthSetter.Size = new System.Drawing.Size(44, 20);
            this.StrengthSetter.TabIndex = 1;
            this.StrengthSetter.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.StrengthSetter.Value = new decimal(new int[] {
            8,
            0,
            0,
            65536});
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(12, 40);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(47, 13);
            this.label1.TabIndex = 2;
            this.label1.Text = "Strength";
            // 
            // ButtonOK
            // 
            this.ButtonOK.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.ButtonOK.Cursor = System.Windows.Forms.Cursors.Default;
            this.ButtonOK.DialogResult = System.Windows.Forms.DialogResult.OK;
            this.ButtonOK.Location = new System.Drawing.Point(11, 113);
            this.ButtonOK.Name = "ButtonOK";
            this.ButtonOK.Size = new System.Drawing.Size(75, 23);
            this.ButtonOK.TabIndex = 4;
            this.ButtonOK.Text = "OK";
            this.ButtonOK.UseVisualStyleBackColor = true;
            // 
            // ButtonCancel
            // 
            this.ButtonCancel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.ButtonCancel.Cursor = System.Windows.Forms.Cursors.Default;
            this.ButtonCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
            this.ButtonCancel.Location = new System.Drawing.Point(92, 113);
            this.ButtonCancel.Name = "ButtonCancel";
            this.ButtonCancel.Size = new System.Drawing.Size(75, 23);
            this.ButtonCancel.TabIndex = 5;
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
            this.PassesSetter.Location = new System.Drawing.Point(123, 12);
            this.PassesSetter.Maximum = new decimal(new int[] {
            10,
            0,
            0,
            0});
            this.PassesSetter.Name = "PassesSetter";
            this.PassesSetter.Size = new System.Drawing.Size(44, 20);
            this.PassesSetter.TabIndex = 0;
            this.PassesSetter.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.PassesSetter.Value = new decimal(new int[] {
            2,
            0,
            0,
            0});
            // 
            // SoftnessSetter
            // 
            this.SoftnessSetter.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.SoftnessSetter.DecimalPlaces = 2;
            this.SoftnessSetter.Increment = new decimal(new int[] {
            5,
            0,
            0,
            131072});
            this.SoftnessSetter.Location = new System.Drawing.Point(123, 64);
            this.SoftnessSetter.Maximum = new decimal(new int[] {
            2,
            0,
            0,
            0});
            this.SoftnessSetter.Name = "SoftnessSetter";
            this.SoftnessSetter.Size = new System.Drawing.Size(44, 20);
            this.SoftnessSetter.TabIndex = 2;
            this.SoftnessSetter.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.SoftnessSetter.Value = new decimal(new int[] {
            5,
            0,
            0,
            65536});
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(12, 66);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(48, 13);
            this.label6.TabIndex = 13;
            this.label6.Text = "Softness";
            // 
            // PrescalerBox
            // 
            this.PrescalerBox.AutoSize = true;
            this.PrescalerBox.Location = new System.Drawing.Point(15, 90);
            this.PrescalerBox.Name = "PrescalerBox";
            this.PrescalerBox.Size = new System.Drawing.Size(130, 17);
            this.PrescalerBox.TabIndex = 3;
            this.PrescalerBox.Text = "Use bilateral prescaler";
            this.PrescalerBox.UseVisualStyleBackColor = true;
            // 
            // SuperChromaResConfigDialog
            // 
            this.AcceptButton = this.ButtonOK;
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.CancelButton = this.ButtonCancel;
            this.ClientSize = new System.Drawing.Size(179, 148);
            this.Controls.Add(this.PrescalerBox);
            this.Controls.Add(this.label6);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.SoftnessSetter);
            this.Controls.Add(this.PassesSetter);
            this.Controls.Add(this.ButtonCancel);
            this.Controls.Add(this.ButtonOK);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.StrengthSetter);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "SuperChromaResConfigDialog";
            this.ShowIcon = false;
            this.ShowInTaskbar = false;
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
            this.Text = "SuperChromaRes Settings";
            ((System.ComponentModel.ISupportInitialize)(this.StrengthSetter)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.PassesSetter)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.SoftnessSetter)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

            }

            #endregion

            private System.Windows.Forms.NumericUpDown StrengthSetter;
            private System.Windows.Forms.Label label1;
            private System.Windows.Forms.Button ButtonOK;
            private System.Windows.Forms.Button ButtonCancel;
            private System.Windows.Forms.Label label5;
            private System.Windows.Forms.NumericUpDown PassesSetter;
            private System.Windows.Forms.NumericUpDown SoftnessSetter;
            private System.Windows.Forms.Label label6;
            private System.Windows.Forms.CheckBox PrescalerBox;

        }
    }
}
