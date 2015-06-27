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

using Mpdn.Extensions.Framework.Controls;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Hylian.SuperXbr
    {
        partial class SuperXbrConfigDialog
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
            this.EdgeStrengthSetter = new System.Windows.Forms.NumericUpDown();
            this.label1 = new System.Windows.Forms.Label();
            this.ButtonOK = new System.Windows.Forms.Button();
            this.ButtonCancel = new System.Windows.Forms.Button();
            this.SharpnessSetter = new System.Windows.Forms.NumericUpDown();
            this.label8 = new System.Windows.Forms.Label();
            this.FastBox = new System.Windows.Forms.CheckBox();
            this.ExtraPassBox = new System.Windows.Forms.CheckBox();
            ((System.ComponentModel.ISupportInitialize)(this.EdgeStrengthSetter)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.SharpnessSetter)).BeginInit();
            this.SuspendLayout();
            // 
            // EdgeStrengthSetter
            // 
            this.EdgeStrengthSetter.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.EdgeStrengthSetter.DecimalPlaces = 2;
            this.EdgeStrengthSetter.Increment = new decimal(new int[] {
            1,
            0,
            0,
            65536});
            this.EdgeStrengthSetter.Location = new System.Drawing.Point(125, 12);
            this.EdgeStrengthSetter.Maximum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.EdgeStrengthSetter.Name = "EdgeStrengthSetter";
            this.EdgeStrengthSetter.Size = new System.Drawing.Size(44, 20);
            this.EdgeStrengthSetter.TabIndex = 1;
            this.EdgeStrengthSetter.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.EdgeStrengthSetter.Value = new decimal(new int[] {
            8,
            0,
            0,
            65536});
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(12, 14);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(75, 13);
            this.label1.TabIndex = 2;
            this.label1.Text = "Edge Strength";
            // 
            // ButtonOK
            // 
            this.ButtonOK.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.ButtonOK.Cursor = System.Windows.Forms.Cursors.Default;
            this.ButtonOK.DialogResult = System.Windows.Forms.DialogResult.OK;
            this.ButtonOK.Location = new System.Drawing.Point(13, 110);
            this.ButtonOK.Name = "ButtonOK";
            this.ButtonOK.Size = new System.Drawing.Size(75, 23);
            this.ButtonOK.TabIndex = 3;
            this.ButtonOK.Text = "OK";
            this.ButtonOK.UseVisualStyleBackColor = true;
            // 
            // ButtonCancel
            // 
            this.ButtonCancel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.ButtonCancel.Cursor = System.Windows.Forms.Cursors.Default;
            this.ButtonCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
            this.ButtonCancel.Location = new System.Drawing.Point(94, 110);
            this.ButtonCancel.Name = "ButtonCancel";
            this.ButtonCancel.Size = new System.Drawing.Size(75, 23);
            this.ButtonCancel.TabIndex = 4;
            this.ButtonCancel.Text = "Cancel";
            this.ButtonCancel.UseVisualStyleBackColor = true;
            // 
            // SharpnessSetter
            // 
            this.SharpnessSetter.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.SharpnessSetter.DecimalPlaces = 2;
            this.SharpnessSetter.Increment = new decimal(new int[] {
            5,
            0,
            0,
            131072});
            this.SharpnessSetter.Location = new System.Drawing.Point(125, 38);
            this.SharpnessSetter.Maximum = new decimal(new int[] {
            2,
            0,
            0,
            0});
            this.SharpnessSetter.Name = "SharpnessSetter";
            this.SharpnessSetter.Size = new System.Drawing.Size(44, 20);
            this.SharpnessSetter.TabIndex = 2;
            this.SharpnessSetter.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.SharpnessSetter.Value = new decimal(new int[] {
            5,
            0,
            0,
            65536});
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(12, 40);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(57, 13);
            this.label8.TabIndex = 16;
            this.label8.Text = "Sharpness";
            // 
            // FastBox
            // 
            this.FastBox.AutoSize = true;
            this.FastBox.Location = new System.Drawing.Point(15, 64);
            this.FastBox.Name = "FastBox";
            this.FastBox.Size = new System.Drawing.Size(84, 17);
            this.FastBox.TabIndex = 17;
            this.FastBox.Text = "Fast method";
            this.FastBox.UseVisualStyleBackColor = true;
            // 
            // ExtraPassBox
            // 
            this.ExtraPassBox.AutoSize = true;
            this.ExtraPassBox.Location = new System.Drawing.Point(15, 87);
            this.ExtraPassBox.Name = "ExtraPassBox";
            this.ExtraPassBox.Size = new System.Drawing.Size(93, 17);
            this.ExtraPassBox.TabIndex = 18;
            this.ExtraPassBox.Text = "Use third pass";
            this.ExtraPassBox.UseVisualStyleBackColor = true;
            // 
            // SuperXbrConfigDialog
            // 
            this.AcceptButton = this.ButtonOK;
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.CancelButton = this.ButtonCancel;
            this.ClientSize = new System.Drawing.Size(181, 145);
            this.Controls.Add(this.ExtraPassBox);
            this.Controls.Add(this.FastBox);
            this.Controls.Add(this.label8);
            this.Controls.Add(this.SharpnessSetter);
            this.Controls.Add(this.ButtonCancel);
            this.Controls.Add(this.ButtonOK);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.EdgeStrengthSetter);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "SuperXbrConfigDialog";
            this.ShowIcon = false;
            this.ShowInTaskbar = false;
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
            this.Text = "Super-xBR Settings";
            ((System.ComponentModel.ISupportInitialize)(this.EdgeStrengthSetter)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.SharpnessSetter)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

            }

            #endregion

            private System.Windows.Forms.NumericUpDown EdgeStrengthSetter;
            private System.Windows.Forms.Label label1;
            private System.Windows.Forms.Button ButtonOK;
            private System.Windows.Forms.Button ButtonCancel;
            private System.Windows.Forms.NumericUpDown SharpnessSetter;
            private System.Windows.Forms.Label label8;
            private System.Windows.Forms.CheckBox FastBox;
            private System.Windows.Forms.CheckBox ExtraPassBox;

        }
    }
}
