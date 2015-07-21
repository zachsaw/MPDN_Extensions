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
    namespace Mpdn.Jinc2D
    {
        partial class Jinc2DConfigDialog
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
            this.AntiRingingStrengthSetter = new System.Windows.Forms.NumericUpDown();
            this.label1 = new System.Windows.Forms.Label();
            this.checkBoxAntiRinging = new System.Windows.Forms.CheckBox();
            this.comboBoxTapCount = new System.Windows.Forms.ComboBox();
            this.label2 = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.AntiRingingStrengthSetter)).BeginInit();
            this.SuspendLayout();
            // 
            // ButtonOK
            // 
            this.ButtonOK.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.ButtonOK.Cursor = System.Windows.Forms.Cursors.Default;
            this.ButtonOK.DialogResult = System.Windows.Forms.DialogResult.OK;
            this.ButtonOK.Location = new System.Drawing.Point(50, 116);
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
            this.ButtonCancel.Location = new System.Drawing.Point(131, 116);
            this.ButtonCancel.Name = "ButtonCancel";
            this.ButtonCancel.Size = new System.Drawing.Size(75, 23);
            this.ButtonCancel.TabIndex = 11;
            this.ButtonCancel.Text = "Cancel";
            this.ButtonCancel.UseVisualStyleBackColor = true;
            // 
            // AntiRingingStrengthSetter
            // 
            this.AntiRingingStrengthSetter.DecimalPlaces = 2;
            this.AntiRingingStrengthSetter.Increment = new decimal(new int[] {
            1,
            0,
            0,
            131072});
            this.AntiRingingStrengthSetter.Location = new System.Drawing.Point(128, 72);
            this.AntiRingingStrengthSetter.Maximum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.AntiRingingStrengthSetter.Name = "AntiRingingStrengthSetter";
            this.AntiRingingStrengthSetter.Size = new System.Drawing.Size(44, 20);
            this.AntiRingingStrengthSetter.TabIndex = 2;
            this.AntiRingingStrengthSetter.Value = new decimal(new int[] {
            85,
            0,
            0,
            131072});
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(73, 74);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(50, 13);
            this.label1.TabIndex = 7;
            this.label1.Text = "Strength:";
            // 
            // checkBoxAntiRinging
            // 
            this.checkBoxAntiRinging.AutoSize = true;
            this.checkBoxAntiRinging.Location = new System.Drawing.Point(57, 46);
            this.checkBoxAntiRinging.Name = "checkBoxAntiRinging";
            this.checkBoxAntiRinging.Size = new System.Drawing.Size(119, 17);
            this.checkBoxAntiRinging.TabIndex = 1;
            this.checkBoxAntiRinging.Text = "Activate anti-ringing";
            this.checkBoxAntiRinging.UseVisualStyleBackColor = true;
            // 
            // comboBoxTapCount
            // 
            this.comboBoxTapCount.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboBoxTapCount.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.comboBoxTapCount.FormattingEnabled = true;
            this.comboBoxTapCount.Location = new System.Drawing.Point(57, 12);
            this.comboBoxTapCount.Name = "comboBoxTapCount";
            this.comboBoxTapCount.Size = new System.Drawing.Size(121, 21);
            this.comboBoxTapCount.TabIndex = 0;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(17, 15);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(34, 13);
            this.label2.TabIndex = 9;
            this.label2.Text = "Taps:";
            // 
            // Jinc2DConfigDialog
            // 
            this.AcceptButton = this.ButtonOK;
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.CancelButton = this.ButtonCancel;
            this.ClientSize = new System.Drawing.Size(218, 151);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.comboBoxTapCount);
            this.Controls.Add(this.checkBoxAntiRinging);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.AntiRingingStrengthSetter);
            this.Controls.Add(this.ButtonCancel);
            this.Controls.Add(this.ButtonOK);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "Jinc2DConfigDialog";
            this.ShowIcon = false;
            this.ShowInTaskbar = false;
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
            this.Text = "Jinc2D Settings";
            ((System.ComponentModel.ISupportInitialize)(this.AntiRingingStrengthSetter)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

            }

            #endregion

            private System.Windows.Forms.Button ButtonOK;
            private System.Windows.Forms.Button ButtonCancel;
            private System.Windows.Forms.NumericUpDown AntiRingingStrengthSetter;
            private System.Windows.Forms.Label label1;
            private System.Windows.Forms.CheckBox checkBoxAntiRinging;
            private System.Windows.Forms.ComboBox comboBoxTapCount;
            private System.Windows.Forms.Label label2;

        }
    }
}
