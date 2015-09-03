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
namespace Mpdn.Extensions.AudioScripts
{
    namespace Mpdn
    {
        partial class DynamicRangeCompressorConfigDialog
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
                this.buttonCancel = new System.Windows.Forms.Button();
                this.buttonOk = new System.Windows.Forms.Button();
                this.textBoxThreshold = new System.Windows.Forms.NumericUpDown();
                this.label1 = new System.Windows.Forms.Label();
                this.label2 = new System.Windows.Forms.Label();
                this.textBoxRatio = new System.Windows.Forms.NumericUpDown();
                this.label3 = new System.Windows.Forms.Label();
                this.textBoxGain = new System.Windows.Forms.NumericUpDown();
                this.labelAttack = new System.Windows.Forms.Label();
                this.textBoxAttack = new System.Windows.Forms.NumericUpDown();
                this.labelRelease = new System.Windows.Forms.Label();
                this.textBoxRelease = new System.Windows.Forms.NumericUpDown();
                ((System.ComponentModel.ISupportInitialize) (this.textBoxThreshold)).BeginInit();
                ((System.ComponentModel.ISupportInitialize) (this.textBoxRatio)).BeginInit();
                ((System.ComponentModel.ISupportInitialize) (this.textBoxGain)).BeginInit();
                ((System.ComponentModel.ISupportInitialize) (this.textBoxAttack)).BeginInit();
                ((System.ComponentModel.ISupportInitialize) (this.textBoxRelease)).BeginInit();
                this.SuspendLayout();
                // 
                // buttonCancel
                // 
                this.buttonCancel.Anchor =
                    ((System.Windows.Forms.AnchorStyles)
                        ((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
                this.buttonCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
                this.buttonCancel.Location = new System.Drawing.Point(122, 158);
                this.buttonCancel.Name = "buttonCancel";
                this.buttonCancel.Size = new System.Drawing.Size(75, 23);
                this.buttonCancel.TabIndex = 1005;
                this.buttonCancel.Text = "Cancel";
                this.buttonCancel.UseVisualStyleBackColor = true;
                // 
                // buttonOk
                // 
                this.buttonOk.Anchor =
                    ((System.Windows.Forms.AnchorStyles)
                        ((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
                this.buttonOk.DialogResult = System.Windows.Forms.DialogResult.OK;
                this.buttonOk.Location = new System.Drawing.Point(41, 158);
                this.buttonOk.Name = "buttonOk";
                this.buttonOk.Size = new System.Drawing.Size(75, 23);
                this.buttonOk.TabIndex = 1004;
                this.buttonOk.Text = "OK";
                this.buttonOk.UseVisualStyleBackColor = true;
                // 
                // textBoxThreshold
                // 
                this.textBoxThreshold.DecimalPlaces = 1;
                this.textBoxThreshold.Increment = new decimal(new int[]
                {
                    5,
                    0,
                    0,
                    65536
                });
                this.textBoxThreshold.Location = new System.Drawing.Point(109, 16);
                this.textBoxThreshold.Maximum = new decimal(new int[]
                {
                    0,
                    0,
                    0,
                    0
                });
                this.textBoxThreshold.Minimum = new decimal(new int[]
                {
                    1000000,
                    0,
                    0,
                    -2147483648
                });
                this.textBoxThreshold.Name = "textBoxThreshold";
                this.textBoxThreshold.Size = new System.Drawing.Size(66, 20);
                this.textBoxThreshold.TabIndex = 0;
                // 
                // label1
                // 
                this.label1.AutoSize = true;
                this.label1.Location = new System.Drawing.Point(21, 19);
                this.label1.Name = "label1";
                this.label1.Size = new System.Drawing.Size(79, 13);
                this.label1.TabIndex = 1007;
                this.label1.Text = "Threshold (dB):";
                // 
                // label2
                // 
                this.label2.AutoSize = true;
                this.label2.Location = new System.Drawing.Point(21, 45);
                this.label2.Name = "label2";
                this.label2.Size = new System.Drawing.Size(35, 13);
                this.label2.TabIndex = 1009;
                this.label2.Text = "Ratio:";
                // 
                // textBoxRatio
                // 
                this.textBoxRatio.DecimalPlaces = 1;
                this.textBoxRatio.Increment = new decimal(new int[]
                {
                    1,
                    0,
                    0,
                    65536
                });
                this.textBoxRatio.Location = new System.Drawing.Point(109, 42);
                this.textBoxRatio.Maximum = new decimal(new int[]
                {
                    20,
                    0,
                    0,
                    0
                });
                this.textBoxRatio.Minimum = new decimal(new int[]
                {
                    1,
                    0,
                    0,
                    0
                });
                this.textBoxRatio.Name = "textBoxRatio";
                this.textBoxRatio.Size = new System.Drawing.Size(66, 20);
                this.textBoxRatio.TabIndex = 1;
                this.textBoxRatio.Value = new decimal(new int[]
                {
                    1,
                    0,
                    0,
                    0
                });
                // 
                // label3
                // 
                this.label3.AutoSize = true;
                this.label3.Location = new System.Drawing.Point(21, 71);
                this.label3.Name = "label3";
                this.label3.Size = new System.Drawing.Size(54, 13);
                this.label3.TabIndex = 1011;
                this.label3.Text = "Gain (dB):";
                // 
                // textBoxGain
                // 
                this.textBoxGain.DecimalPlaces = 1;
                this.textBoxGain.Increment = new decimal(new int[]
                {
                    5,
                    0,
                    0,
                    65536
                });
                this.textBoxGain.Location = new System.Drawing.Point(109, 68);
                this.textBoxGain.Name = "textBoxGain";
                this.textBoxGain.Size = new System.Drawing.Size(66, 20);
                this.textBoxGain.TabIndex = 2;
                // 
                // labelAttack
                // 
                this.labelAttack.AutoSize = true;
                this.labelAttack.Location = new System.Drawing.Point(21, 97);
                this.labelAttack.Name = "labelAttack";
                this.labelAttack.Size = new System.Drawing.Size(55, 13);
                this.labelAttack.TabIndex = 1013;
                this.labelAttack.Text = "Attack (s):";
                // 
                // textBoxAttack
                // 
                this.textBoxAttack.DecimalPlaces = 1;
                this.textBoxAttack.Increment = new decimal(new int[]
                {
                    1,
                    0,
                    0,
                    65536
                });
                this.textBoxAttack.Location = new System.Drawing.Point(109, 94);
                this.textBoxAttack.Maximum = new decimal(new int[]
                {
                    5,
                    0,
                    0,
                    0
                });
                this.textBoxAttack.Minimum = new decimal(new int[]
                {
                    1,
                    0,
                    0,
                    65536
                });
                this.textBoxAttack.Name = "textBoxAttack";
                this.textBoxAttack.Size = new System.Drawing.Size(66, 20);
                this.textBoxAttack.TabIndex = 3;
                this.textBoxAttack.Value = new decimal(new int[]
                {
                    1,
                    0,
                    0,
                    65536
                });
                // 
                // labelRelease
                // 
                this.labelRelease.AutoSize = true;
                this.labelRelease.Location = new System.Drawing.Point(21, 123);
                this.labelRelease.Name = "labelRelease";
                this.labelRelease.Size = new System.Drawing.Size(63, 13);
                this.labelRelease.TabIndex = 1015;
                this.labelRelease.Text = "Release (s):";
                // 
                // textBoxRelease
                // 
                this.textBoxRelease.DecimalPlaces = 1;
                this.textBoxRelease.Increment = new decimal(new int[]
                {
                    1,
                    0,
                    0,
                    65536
                });
                this.textBoxRelease.Location = new System.Drawing.Point(109, 120);
                this.textBoxRelease.Maximum = new decimal(new int[]
                {
                    5,
                    0,
                    0,
                    0
                });
                this.textBoxRelease.Minimum = new decimal(new int[]
                {
                    1,
                    0,
                    0,
                    65536
                });
                this.textBoxRelease.Name = "textBoxRelease";
                this.textBoxRelease.Size = new System.Drawing.Size(66, 20);
                this.textBoxRelease.TabIndex = 4;
                this.textBoxRelease.Value = new decimal(new int[]
                {
                    1,
                    0,
                    0,
                    65536
                });
                // 
                // DynamicRangeCompressorConfigDialog
                // 
                this.AcceptButton = this.buttonOk;
                this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
                this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
                this.CancelButton = this.buttonCancel;
                this.ClientSize = new System.Drawing.Size(209, 193);
                this.Controls.Add(this.labelRelease);
                this.Controls.Add(this.textBoxRelease);
                this.Controls.Add(this.labelAttack);
                this.Controls.Add(this.textBoxAttack);
                this.Controls.Add(this.label3);
                this.Controls.Add(this.textBoxGain);
                this.Controls.Add(this.label2);
                this.Controls.Add(this.textBoxRatio);
                this.Controls.Add(this.label1);
                this.Controls.Add(this.textBoxThreshold);
                this.Controls.Add(this.buttonCancel);
                this.Controls.Add(this.buttonOk);
                this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
                this.MaximizeBox = false;
                this.MinimizeBox = false;
                this.Name = "DynamicRangeCompressorConfigDialog";
                this.ShowInTaskbar = false;
                this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
                this.Text = "DRC Configuration";
                ((System.ComponentModel.ISupportInitialize) (this.textBoxThreshold)).EndInit();
                ((System.ComponentModel.ISupportInitialize) (this.textBoxRatio)).EndInit();
                ((System.ComponentModel.ISupportInitialize) (this.textBoxGain)).EndInit();
                ((System.ComponentModel.ISupportInitialize) (this.textBoxAttack)).EndInit();
                ((System.ComponentModel.ISupportInitialize) (this.textBoxRelease)).EndInit();
                this.ResumeLayout(false);
                this.PerformLayout();

            }

            #endregion

            private System.Windows.Forms.Button buttonCancel;
            private System.Windows.Forms.Button buttonOk;
            private System.Windows.Forms.NumericUpDown textBoxThreshold;
            private System.Windows.Forms.Label label1;
            private System.Windows.Forms.Label label2;
            private System.Windows.Forms.NumericUpDown textBoxRatio;
            private System.Windows.Forms.Label label3;
            private System.Windows.Forms.NumericUpDown textBoxGain;
            private System.Windows.Forms.Label labelAttack;
            private System.Windows.Forms.NumericUpDown textBoxAttack;
            private System.Windows.Forms.Label labelRelease;
            private System.Windows.Forms.NumericUpDown textBoxRelease;
        }
    }
}