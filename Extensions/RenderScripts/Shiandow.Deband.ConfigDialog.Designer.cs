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
    namespace Shiandow.Deband
    {
        partial class DebandConfigDialog
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
            this.ButtonOK = new System.Windows.Forms.Button();
            this.ButtonCancel = new System.Windows.Forms.Button();
            this.label5 = new System.Windows.Forms.Label();
            this.MaxBitdepthSetter = new System.Windows.Forms.NumericUpDown();
            this.label3 = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.PowerSetter = new System.Windows.Forms.NumericUpDown();
            this.GrainBox = new System.Windows.Forms.CheckBox();
            ((System.ComponentModel.ISupportInitialize)(this.MaxBitdepthSetter)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.PowerSetter)).BeginInit();
            this.SuspendLayout();
            // 
            // ButtonOK
            // 
            this.ButtonOK.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.ButtonOK.Cursor = System.Windows.Forms.Cursors.Default;
            this.ButtonOK.DialogResult = System.Windows.Forms.DialogResult.OK;
            this.ButtonOK.Location = new System.Drawing.Point(70, 95);
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
            this.ButtonCancel.Location = new System.Drawing.Point(151, 95);
            this.ButtonCancel.Name = "ButtonCancel";
            this.ButtonCancel.Size = new System.Drawing.Size(75, 23);
            this.ButtonCancel.TabIndex = 4;
            this.ButtonCancel.Text = "Cancel";
            this.ButtonCancel.UseVisualStyleBackColor = true;
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(20, 15);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(121, 13);
            this.label5.TabIndex = 11;
            this.label5.Text = "Disable when bitdepth >";
            // 
            // MaxBitdepthSetter
            // 
            this.MaxBitdepthSetter.Location = new System.Drawing.Point(153, 13);
            this.MaxBitdepthSetter.Maximum = new decimal(new int[] {
            32,
            0,
            0,
            0});
            this.MaxBitdepthSetter.Minimum = new decimal(new int[] {
            8,
            0,
            0,
            0});
            this.MaxBitdepthSetter.Name = "MaxBitdepthSetter";
            this.MaxBitdepthSetter.Size = new System.Drawing.Size(44, 20);
            this.MaxBitdepthSetter.TabIndex = 0;
            this.MaxBitdepthSetter.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.MaxBitdepthSetter.Value = new decimal(new int[] {
            8,
            0,
            0,
            0});
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(203, 15);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(23, 13);
            this.label3.TabIndex = 12;
            this.label3.Text = "bits";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(20, 41);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(37, 13);
            this.label4.TabIndex = 17;
            this.label4.Text = "Power";
            // 
            // PowerSetter
            // 
            this.PowerSetter.DecimalPlaces = 2;
            this.PowerSetter.Increment = new decimal(new int[] {
            1,
            0,
            0,
            131072});
            this.PowerSetter.Location = new System.Drawing.Point(153, 39);
            this.PowerSetter.Maximum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.PowerSetter.Name = "PowerSetter";
            this.PowerSetter.Size = new System.Drawing.Size(44, 20);
            this.PowerSetter.TabIndex = 16;
            this.PowerSetter.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.PowerSetter.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
            // 
            // GrainBox
            // 
            this.GrainBox.AutoSize = true;
            this.GrainBox.Location = new System.Drawing.Point(153, 65);
            this.GrainBox.Name = "GrainBox";
            this.GrainBox.Size = new System.Drawing.Size(71, 17);
            this.GrainBox.TabIndex = 18;
            this.GrainBox.Text = "Add grain";
            this.GrainBox.UseVisualStyleBackColor = true;
            // 
            // DebandConfigDialog
            // 
            this.AcceptButton = this.ButtonOK;
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.CancelButton = this.ButtonCancel;
            this.ClientSize = new System.Drawing.Size(238, 130);
            this.Controls.Add(this.GrainBox);
            this.Controls.Add(this.PowerSetter);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.MaxBitdepthSetter);
            this.Controls.Add(this.ButtonCancel);
            this.Controls.Add(this.ButtonOK);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "DebandConfigDialog";
            this.ShowIcon = false;
            this.ShowInTaskbar = false;
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
            this.Text = "Deband Settings";
            ((System.ComponentModel.ISupportInitialize)(this.MaxBitdepthSetter)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.PowerSetter)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

            }

            #endregion

            private System.Windows.Forms.Button ButtonOK;
            private System.Windows.Forms.Button ButtonCancel;
            private System.Windows.Forms.Label label5;
            private System.Windows.Forms.NumericUpDown MaxBitdepthSetter;
            private System.Windows.Forms.Label label3;
            private System.Windows.Forms.Label label4;
            private System.Windows.Forms.NumericUpDown PowerSetter;
            private System.Windows.Forms.CheckBox GrainBox;

        }
    }
}
