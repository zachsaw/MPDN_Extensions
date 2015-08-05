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
    namespace Mpdn.EwaScaler
    {
        partial class EwaScalerConfigDialog
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
            this.setterAntiRingStrength = new System.Windows.Forms.NumericUpDown();
            this.labelStrength = new System.Windows.Forms.Label();
            this.checkBoxAntiRinging = new System.Windows.Forms.CheckBox();
            this.comboBoxTapCount = new System.Windows.Forms.ComboBox();
            this.label2 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.comboBoxScaler = new System.Windows.Forms.ComboBox();
            ((System.ComponentModel.ISupportInitialize)(this.setterAntiRingStrength)).BeginInit();
            this.SuspendLayout();
            // 
            // ButtonOK
            // 
            this.ButtonOK.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.ButtonOK.Cursor = System.Windows.Forms.Cursors.Default;
            this.ButtonOK.DialogResult = System.Windows.Forms.DialogResult.OK;
            this.ButtonOK.Location = new System.Drawing.Point(55, 159);
            this.ButtonOK.Name = "ButtonOK";
            this.ButtonOK.Size = new System.Drawing.Size(75, 23);
            this.ButtonOK.TabIndex = 10;
            this.ButtonOK.Text = "OK";
            this.ButtonOK.UseVisualStyleBackColor = true;
            // 
            // ButtonCancel
            // 
            this.ButtonCancel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.ButtonCancel.Cursor = System.Windows.Forms.Cursors.Default;
            this.ButtonCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
            this.ButtonCancel.Location = new System.Drawing.Point(136, 159);
            this.ButtonCancel.Name = "ButtonCancel";
            this.ButtonCancel.Size = new System.Drawing.Size(75, 23);
            this.ButtonCancel.TabIndex = 11;
            this.ButtonCancel.Text = "Cancel";
            this.ButtonCancel.UseVisualStyleBackColor = true;
            // 
            // setterAntiRingStrength
            // 
            this.setterAntiRingStrength.DecimalPlaces = 2;
            this.setterAntiRingStrength.Increment = new decimal(new int[] {
            15,
            0,
            0,
            131072});
            this.setterAntiRingStrength.Location = new System.Drawing.Point(135, 113);
            this.setterAntiRingStrength.Maximum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.setterAntiRingStrength.Name = "setterAntiRingStrength";
            this.setterAntiRingStrength.Size = new System.Drawing.Size(44, 20);
            this.setterAntiRingStrength.TabIndex = 3;
            this.setterAntiRingStrength.Value = new decimal(new int[] {
            85,
            0,
            0,
            131072});
            // 
            // labelStrength
            // 
            this.labelStrength.AutoSize = true;
            this.labelStrength.Location = new System.Drawing.Point(79, 115);
            this.labelStrength.Name = "labelStrength";
            this.labelStrength.Size = new System.Drawing.Size(50, 13);
            this.labelStrength.TabIndex = 7;
            this.labelStrength.Text = "Strength:";
            // 
            // checkBoxAntiRinging
            // 
            this.checkBoxAntiRinging.AutoSize = true;
            this.checkBoxAntiRinging.Location = new System.Drawing.Point(64, 87);
            this.checkBoxAntiRinging.Name = "checkBoxAntiRinging";
            this.checkBoxAntiRinging.Size = new System.Drawing.Size(119, 17);
            this.checkBoxAntiRinging.TabIndex = 2;
            this.checkBoxAntiRinging.Text = "Activate anti-ringing";
            this.checkBoxAntiRinging.UseVisualStyleBackColor = true;
            // 
            // comboBoxTapCount
            // 
            this.comboBoxTapCount.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboBoxTapCount.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.comboBoxTapCount.FormattingEnabled = true;
            this.comboBoxTapCount.Location = new System.Drawing.Point(64, 54);
            this.comboBoxTapCount.Name = "comboBoxTapCount";
            this.comboBoxTapCount.Size = new System.Drawing.Size(121, 21);
            this.comboBoxTapCount.TabIndex = 1;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(24, 57);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(34, 13);
            this.label2.TabIndex = 9;
            this.label2.Text = "Taps:";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(18, 24);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(40, 13);
            this.label3.TabIndex = 13;
            this.label3.Text = "Scaler:";
            // 
            // comboBoxScaler
            // 
            this.comboBoxScaler.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboBoxScaler.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.comboBoxScaler.FormattingEnabled = true;
            this.comboBoxScaler.Location = new System.Drawing.Point(64, 21);
            this.comboBoxScaler.Name = "comboBoxScaler";
            this.comboBoxScaler.Size = new System.Drawing.Size(121, 21);
            this.comboBoxScaler.TabIndex = 0;
            this.comboBoxScaler.SelectedIndexChanged += new System.EventHandler(this.ScalerSelectionChanged);
            // 
            // EwaScalerConfigDialog
            // 
            this.AcceptButton = this.ButtonOK;
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.CancelButton = this.ButtonCancel;
            this.ClientSize = new System.Drawing.Size(223, 194);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.comboBoxScaler);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.comboBoxTapCount);
            this.Controls.Add(this.checkBoxAntiRinging);
            this.Controls.Add(this.labelStrength);
            this.Controls.Add(this.setterAntiRingStrength);
            this.Controls.Add(this.ButtonCancel);
            this.Controls.Add(this.ButtonOK);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "EwaScalerConfigDialog";
            this.ShowIcon = false;
            this.ShowInTaskbar = false;
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
            this.Text = "EwaScaler Settings";
            ((System.ComponentModel.ISupportInitialize)(this.setterAntiRingStrength)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

            }

            #endregion

            private System.Windows.Forms.Button ButtonOK;
            private System.Windows.Forms.Button ButtonCancel;
            private System.Windows.Forms.NumericUpDown setterAntiRingStrength;
            private System.Windows.Forms.Label labelStrength;
            private System.Windows.Forms.CheckBox checkBoxAntiRinging;
            private System.Windows.Forms.ComboBox comboBoxTapCount;
            private System.Windows.Forms.Label label2;
            private System.Windows.Forms.Label label3;
            private System.Windows.Forms.ComboBox comboBoxScaler;

        }
    }
}
