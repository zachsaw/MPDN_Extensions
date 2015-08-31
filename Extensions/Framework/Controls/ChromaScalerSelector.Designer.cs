namespace Mpdn.Extensions.Framework.Controls
{
    partial class ChromaScalerSelector
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

        #region Component Designer generated code

        /// <summary> 
        /// Required method for Designer support - do not modify 
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.buttonConfig = new System.Windows.Forms.Button();
            this.comboBoxChroma = new System.Windows.Forms.ComboBox();
            this.SuspendLayout();
            // 
            // buttonConfig
            // 
            this.buttonConfig.Enabled = false;
            this.buttonConfig.Location = new System.Drawing.Point(3, 30);
            this.buttonConfig.Name = "buttonConfig";
            this.buttonConfig.Size = new System.Drawing.Size(75, 23);
            this.buttonConfig.TabIndex = 1;
            this.buttonConfig.Text = "Configure...";
            this.buttonConfig.UseVisualStyleBackColor = true;
            this.buttonConfig.Click += new System.EventHandler(this.ButtonConfigClick);
            // 
            // comboBoxChroma
            // 
            this.comboBoxChroma.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboBoxChroma.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.comboBoxChroma.FormattingEnabled = true;
            this.comboBoxChroma.Location = new System.Drawing.Point(3, 3);
            this.comboBoxChroma.Name = "comboBoxChroma";
            this.comboBoxChroma.Size = new System.Drawing.Size(176, 21);
            this.comboBoxChroma.TabIndex = 0;
            this.comboBoxChroma.SelectedIndexChanged += new System.EventHandler(this.ChromaSelectedIndexChanged);
            // 
            // ChromaScalerSelector
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.buttonConfig);
            this.Controls.Add(this.comboBoxChroma);
            this.Name = "ChromaScalerSelector";
            this.Size = new System.Drawing.Size(182, 59);
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Button buttonConfig;
        private System.Windows.Forms.ComboBox comboBoxChroma;
    }
}
