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
    partial class GoToForm
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
            this.btn_timeOk = new System.Windows.Forms.Button();
            this.tb_time = new System.Windows.Forms.MaskedTextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.btn_frameOk = new System.Windows.Forms.Button();
            this.label3 = new System.Windows.Forms.Label();
            this.nud_frame = new System.Windows.Forms.NumericUpDown();
            ((System.ComponentModel.ISupportInitialize)(this.nud_frame)).BeginInit();
            this.SuspendLayout();
            // 
            // btn_timeOk
            // 
            this.btn_timeOk.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.btn_timeOk.Cursor = System.Windows.Forms.Cursors.Default;
            this.btn_timeOk.DialogResult = System.Windows.Forms.DialogResult.OK;
            this.btn_timeOk.Location = new System.Drawing.Point(161, 16);
            this.btn_timeOk.Name = "btn_timeOk";
            this.btn_timeOk.Size = new System.Drawing.Size(38, 26);
            this.btn_timeOk.TabIndex = 3;
            this.btn_timeOk.TabStop = false;
            this.btn_timeOk.Text = "Go";
            this.btn_timeOk.UseVisualStyleBackColor = true;
            this.btn_timeOk.Click += new System.EventHandler(this.ButtonTimeOkClick);
            // 
            // tb_time
            // 
            this.tb_time.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F);
            this.tb_time.InsertKeyMode = System.Windows.Forms.InsertKeyMode.Overwrite;
            this.tb_time.Location = new System.Drawing.Point(58, 19);
            this.tb_time.Mask = "00:00:00.000";
            this.tb_time.Name = "tb_time";
            this.tb_time.PromptChar = '0';
            this.tb_time.ResetOnPrompt = false;
            this.tb_time.Size = new System.Drawing.Size(97, 20);
            this.tb_time.TabIndex = 0;
            this.tb_time.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            this.tb_time.TextMaskFormat = System.Windows.Forms.MaskFormat.IncludePromptAndLiterals;
            this.tb_time.KeyDown += new System.Windows.Forms.KeyEventHandler(this.TbTimeKeyDown);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(25, 22);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(33, 13);
            this.label1.TabIndex = 6;
            this.label1.Text = "Time:";
            // 
            // label2
            // 
            this.label2.Font = new System.Drawing.Font("Consolas", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label2.ForeColor = System.Drawing.SystemColors.AppWorkspace;
            this.label2.Location = new System.Drawing.Point(59, 38);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(97, 14);
            this.label2.TabIndex = 7;
            this.label2.Text = "(hh:mm:ss.msec)";
            this.label2.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // btn_frameOk
            // 
            this.btn_frameOk.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.btn_frameOk.Cursor = System.Windows.Forms.Cursors.Default;
            this.btn_frameOk.DialogResult = System.Windows.Forms.DialogResult.OK;
            this.btn_frameOk.Location = new System.Drawing.Point(161, 62);
            this.btn_frameOk.Name = "btn_frameOk";
            this.btn_frameOk.Size = new System.Drawing.Size(38, 26);
            this.btn_frameOk.TabIndex = 9;
            this.btn_frameOk.TabStop = false;
            this.btn_frameOk.Text = "Go";
            this.btn_frameOk.UseVisualStyleBackColor = true;
            this.btn_frameOk.Click += new System.EventHandler(this.ButtonFrameOkClick);
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(19, 68);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(39, 13);
            this.label3.TabIndex = 10;
            this.label3.Text = "Frame:";
            // 
            // nud_frame
            // 
            this.nud_frame.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F);
            this.nud_frame.Location = new System.Drawing.Point(58, 65);
            this.nud_frame.Name = "nud_frame";
            this.nud_frame.Size = new System.Drawing.Size(97, 20);
            this.nud_frame.TabIndex = 2;
            this.nud_frame.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            this.nud_frame.Enter += new System.EventHandler(this.NudFrameEnter);
            this.nud_frame.KeyDown += new System.Windows.Forms.KeyEventHandler(this.NudFrameKeyDown);
            // 
            // GoToForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(218, 102);
            this.Controls.Add(this.btn_frameOk);
            this.Controls.Add(this.nud_frame);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.btn_timeOk);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.tb_time);
            this.Controls.Add(this.label2);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.KeyPreview = true;
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "GoToForm";
            this.ShowInTaskbar = false;
            this.StartPosition = System.Windows.Forms.FormStartPosition.Manual;
            this.Text = "Go To";
            ((System.ComponentModel.ISupportInitialize)(this.nud_frame)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button btn_timeOk;
        private System.Windows.Forms.MaskedTextBox tb_time;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Button btn_frameOk;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.NumericUpDown nud_frame;
    }
}
