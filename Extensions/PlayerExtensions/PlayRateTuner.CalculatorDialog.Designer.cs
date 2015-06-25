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
    namespace GitHub
    {
        partial class PlayrateTunerCalculatorDialog
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
            this.buttonOk = new System.Windows.Forms.Button();
            this.textBoxVsyncHz = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.textBoxFrameRateHz = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.textBoxRefClk = new System.Windows.Forms.TextBox();
            this.label6 = new System.Windows.Forms.Label();
            this.label7 = new System.Windows.Forms.Label();
            this.textBoxAnswer = new System.Windows.Forms.TextBox();
            this.label8 = new System.Windows.Forms.Label();
            this.linkCopy = new System.Windows.Forms.LinkLabel();
            this.SuspendLayout();
            // 
            // buttonOk
            // 
            this.buttonOk.Anchor = System.Windows.Forms.AnchorStyles.Bottom;
            this.buttonOk.DialogResult = System.Windows.Forms.DialogResult.OK;
            this.buttonOk.Location = new System.Drawing.Point(119, 135);
            this.buttonOk.Name = "buttonOk";
            this.buttonOk.Size = new System.Drawing.Size(75, 23);
            this.buttonOk.TabIndex = 1004;
            this.buttonOk.Text = "&Close";
            this.buttonOk.UseVisualStyleBackColor = true;
            // 
            // textBoxVsyncHz
            // 
            this.textBoxVsyncHz.Location = new System.Drawing.Point(128, 12);
            this.textBoxVsyncHz.Name = "textBoxVsyncHz";
            this.textBoxVsyncHz.Size = new System.Drawing.Size(150, 20);
            this.textBoxVsyncHz.TabIndex = 0;
            this.textBoxVsyncHz.TextChanged += new System.EventHandler(this.InputTextChanged);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(12, 15);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(110, 13);
            this.label1.TabIndex = 1005;
            this.label1.Text = "Display Refresh Rate:";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(280, 15);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(20, 13);
            this.label2.TabIndex = 1006;
            this.label2.Text = "Hz";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(12, 41);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(95, 13);
            this.label3.TabIndex = 1007;
            this.label3.Text = "Video Frame Rate:";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(280, 41);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(20, 13);
            this.label4.TabIndex = 1009;
            this.label4.Text = "Hz";
            // 
            // textBoxFrameRateHz
            // 
            this.textBoxFrameRateHz.Location = new System.Drawing.Point(128, 38);
            this.textBoxFrameRateHz.Name = "textBoxFrameRateHz";
            this.textBoxFrameRateHz.Size = new System.Drawing.Size(150, 20);
            this.textBoxFrameRateHz.TabIndex = 1;
            this.textBoxFrameRateHz.TextChanged += new System.EventHandler(this.InputTextChanged);
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(280, 67);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(15, 13);
            this.label5.TabIndex = 1012;
            this.label5.Text = "%";
            // 
            // textBoxRefClk
            // 
            this.textBoxRefClk.Location = new System.Drawing.Point(128, 64);
            this.textBoxRefClk.Name = "textBoxRefClk";
            this.textBoxRefClk.Size = new System.Drawing.Size(150, 20);
            this.textBoxRefClk.TabIndex = 3;
            this.textBoxRefClk.TextChanged += new System.EventHandler(this.InputTextChanged);
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(12, 67);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(105, 13);
            this.label6.TabIndex = 1010;
            this.label6.Text = "Ref Clock Deviation:";
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(280, 103);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(15, 13);
            this.label7.TabIndex = 1015;
            this.label7.Text = "%";
            // 
            // textBoxAnswer
            // 
            this.textBoxAnswer.BackColor = System.Drawing.SystemColors.Window;
            this.textBoxAnswer.Location = new System.Drawing.Point(128, 100);
            this.textBoxAnswer.Name = "textBoxAnswer";
            this.textBoxAnswer.ReadOnly = true;
            this.textBoxAnswer.Size = new System.Drawing.Size(150, 20);
            this.textBoxAnswer.TabIndex = 4;
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(12, 103);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(41, 13);
            this.label8.TabIndex = 1014;
            this.label8.Text = "Speed:";
            // 
            // linkCopy
            // 
            this.linkCopy.AutoSize = true;
            this.linkCopy.LinkBehavior = System.Windows.Forms.LinkBehavior.NeverUnderline;
            this.linkCopy.Location = new System.Drawing.Point(54, 103);
            this.linkCopy.Name = "linkCopy";
            this.linkCopy.Size = new System.Drawing.Size(43, 13);
            this.linkCopy.TabIndex = 1016;
            this.linkCopy.TabStop = true;
            this.linkCopy.Text = "[ Copy] ";
            this.linkCopy.LinkClicked += new System.Windows.Forms.LinkLabelLinkClickedEventHandler(this.LinkCopyLinkClicked);
            // 
            // PlayrateTunerCalculatorDialog
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.CancelButton = this.buttonOk;
            this.ClientSize = new System.Drawing.Size(313, 170);
            this.Controls.Add(this.linkCopy);
            this.Controls.Add(this.label7);
            this.Controls.Add(this.textBoxAnswer);
            this.Controls.Add(this.label8);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.textBoxRefClk);
            this.Controls.Add(this.label6);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.textBoxFrameRateHz);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.textBoxVsyncHz);
            this.Controls.Add(this.buttonOk);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "PlayrateTunerCalculatorDialog";
            this.ShowInTaskbar = false;
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
            this.Text = "Speed Calculator";
            this.ResumeLayout(false);
            this.PerformLayout();

            }

            #endregion

            private System.Windows.Forms.Button buttonOk;
            private System.Windows.Forms.TextBox textBoxVsyncHz;
            private System.Windows.Forms.Label label1;
            private System.Windows.Forms.Label label2;
            private System.Windows.Forms.Label label3;
            private System.Windows.Forms.Label label4;
            private System.Windows.Forms.TextBox textBoxFrameRateHz;
            private System.Windows.Forms.Label label5;
            private System.Windows.Forms.TextBox textBoxRefClk;
            private System.Windows.Forms.Label label6;
            private System.Windows.Forms.Label label7;
            private System.Windows.Forms.TextBox textBoxAnswer;
            private System.Windows.Forms.Label label8;
            private System.Windows.Forms.LinkLabel linkCopy;

        }
    }
}
