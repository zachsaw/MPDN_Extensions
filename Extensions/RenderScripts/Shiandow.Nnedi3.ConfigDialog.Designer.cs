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
    namespace Shiandow.NNedi3
    {
        partial class NNedi3ConfigDialog
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
            this.comboBoxNeurons1 = new System.Windows.Forms.ComboBox();
            this.label1 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.comboBoxPath = new System.Windows.Forms.ComboBox();
            this.label3 = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.comboBoxNeurons2 = new System.Windows.Forms.ComboBox();
            this.label5 = new System.Windows.Forms.Label();
            this.label6 = new System.Windows.Forms.Label();
            this.checkBoxStructured = new System.Windows.Forms.CheckBox();
            this.label7 = new System.Windows.Forms.Label();
            this.comboBoxChroma = new System.Windows.Forms.ComboBox();
            this.buttonConfig = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // buttonCancel
            // 
            this.buttonCancel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.buttonCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
            this.buttonCancel.Location = new System.Drawing.Point(193, 376);
            this.buttonCancel.Name = "buttonCancel";
            this.buttonCancel.Size = new System.Drawing.Size(75, 23);
            this.buttonCancel.TabIndex = 1005;
            this.buttonCancel.Text = "Cancel";
            this.buttonCancel.UseVisualStyleBackColor = true;
            // 
            // buttonOk
            // 
            this.buttonOk.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.buttonOk.DialogResult = System.Windows.Forms.DialogResult.OK;
            this.buttonOk.Location = new System.Drawing.Point(112, 376);
            this.buttonOk.Name = "buttonOk";
            this.buttonOk.Size = new System.Drawing.Size(75, 23);
            this.buttonOk.TabIndex = 1004;
            this.buttonOk.Text = "OK";
            this.buttonOk.UseVisualStyleBackColor = true;
            // 
            // comboBoxNeurons1
            // 
            this.comboBoxNeurons1.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboBoxNeurons1.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.comboBoxNeurons1.FormattingEnabled = true;
            this.comboBoxNeurons1.Items.AddRange(new object[] {
            "16",
            "32",
            "64",
            "128",
            "256"});
            this.comboBoxNeurons1.Location = new System.Drawing.Point(100, 43);
            this.comboBoxNeurons1.Name = "comboBoxNeurons1";
            this.comboBoxNeurons1.Size = new System.Drawing.Size(67, 21);
            this.comboBoxNeurons1.TabIndex = 1;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(24, 18);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(54, 13);
            this.label1.TabIndex = 1007;
            this.label1.Text = "Neurons*:";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(24, 148);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(71, 13);
            this.label2.TabIndex = 1009;
            this.label2.Text = "Optimization*:";
            // 
            // comboBoxPath
            // 
            this.comboBoxPath.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboBoxPath.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.comboBoxPath.FormattingEnabled = true;
            this.comboBoxPath.Location = new System.Drawing.Point(64, 199);
            this.comboBoxPath.Name = "comboBoxPath";
            this.comboBoxPath.Size = new System.Drawing.Size(152, 21);
            this.comboBoxPath.TabIndex = 4;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(61, 231);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(192, 26);
            this.label3.TabIndex = 1010;
            this.label3.Text = "* Try all optimization options and select \r\n   the fastest for the chosen neurons" +
    "";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(52, 73);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(42, 13);
            this.label4.TabIndex = 1012;
            this.label4.Text = "Pass 2:";
            // 
            // comboBoxNeurons2
            // 
            this.comboBoxNeurons2.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboBoxNeurons2.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.comboBoxNeurons2.FormattingEnabled = true;
            this.comboBoxNeurons2.Items.AddRange(new object[] {
            "16",
            "32",
            "64",
            "128",
            "256"});
            this.comboBoxNeurons2.Location = new System.Drawing.Point(101, 70);
            this.comboBoxNeurons2.Name = "comboBoxNeurons2";
            this.comboBoxNeurons2.Size = new System.Drawing.Size(67, 21);
            this.comboBoxNeurons2.TabIndex = 2;
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(52, 46);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(42, 13);
            this.label5.TabIndex = 1014;
            this.label5.Text = "Pass 1:";
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(61, 102);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(193, 26);
            this.label6.TabIndex = 1015;
            this.label6.Text = "* Selecting higher neurons for first pass \r\n   gives better quality";
            // 
            // checkBoxStructured
            // 
            this.checkBoxStructured.AutoSize = true;
            this.checkBoxStructured.Location = new System.Drawing.Point(64, 176);
            this.checkBoxStructured.Name = "checkBoxStructured";
            this.checkBoxStructured.Size = new System.Drawing.Size(177, 17);
            this.checkBoxStructured.TabIndex = 3;
            this.checkBoxStructured.Text = "Alternate weight access method";
            this.checkBoxStructured.UseVisualStyleBackColor = true;
            this.checkBoxStructured.CheckedChanged += new System.EventHandler(this.StructuredCheckedChanged);
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(24, 271);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(117, 13);
            this.label7.TabIndex = 1017;
            this.label7.Text = "Custom Chroma Scaler:";
            // 
            // comboBoxChroma
            // 
            this.comboBoxChroma.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboBoxChroma.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.comboBoxChroma.FormattingEnabled = true;
            this.comboBoxChroma.Location = new System.Drawing.Point(64, 298);
            this.comboBoxChroma.Name = "comboBoxChroma";
            this.comboBoxChroma.Size = new System.Drawing.Size(152, 21);
            this.comboBoxChroma.TabIndex = 5;
            this.comboBoxChroma.SelectedIndexChanged += new System.EventHandler(this.ChromaSelectedIndexChanged);
            // 
            // buttonConfig
            // 
            this.buttonConfig.Location = new System.Drawing.Point(64, 325);
            this.buttonConfig.Name = "buttonConfig";
            this.buttonConfig.Size = new System.Drawing.Size(75, 23);
            this.buttonConfig.TabIndex = 6;
            this.buttonConfig.Text = "Configure...";
            this.buttonConfig.UseVisualStyleBackColor = true;
            this.buttonConfig.Click += new System.EventHandler(this.ButtonConfigClick);
            // 
            // NNedi3ConfigDialog
            // 
            this.AcceptButton = this.buttonOk;
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.CancelButton = this.buttonCancel;
            this.ClientSize = new System.Drawing.Size(280, 411);
            this.Controls.Add(this.buttonConfig);
            this.Controls.Add(this.label7);
            this.Controls.Add(this.comboBoxChroma);
            this.Controls.Add(this.checkBoxStructured);
            this.Controls.Add(this.label6);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.comboBoxNeurons2);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.comboBoxPath);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.comboBoxNeurons1);
            this.Controls.Add(this.buttonCancel);
            this.Controls.Add(this.buttonOk);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "NNedi3ConfigDialog";
            this.ShowInTaskbar = false;
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
            this.Text = "NNEDI3 Configuration";
            this.ResumeLayout(false);
            this.PerformLayout();

            }

            #endregion

            private System.Windows.Forms.Button buttonCancel;
            private System.Windows.Forms.Button buttonOk;
            private System.Windows.Forms.ComboBox comboBoxNeurons1;
            private System.Windows.Forms.Label label1;
            private System.Windows.Forms.Label label2;
            private System.Windows.Forms.ComboBox comboBoxPath;
            private System.Windows.Forms.Label label3;
            private System.Windows.Forms.Label label4;
            private System.Windows.Forms.ComboBox comboBoxNeurons2;
            private System.Windows.Forms.Label label5;
            private System.Windows.Forms.Label label6;
            private System.Windows.Forms.CheckBox checkBoxStructured;
            private System.Windows.Forms.Label label7;
            private System.Windows.Forms.ComboBox comboBoxChroma;
            private System.Windows.Forms.Button buttonConfig;
        }
    }
}
