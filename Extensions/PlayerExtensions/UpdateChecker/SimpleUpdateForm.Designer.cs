namespace Mpdn.Extensions.PlayerExtensions.UpdateChecker
{
    partial class SimpleUpdateForm
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
            this.installButton = new System.Windows.Forms.Button();
            this.cancelButton = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.forgetUpdateButton = new System.Windows.Forms.Button();
            this.downloadProgressBar = new Mpdn.Extensions.PlayerExtensions.UpdateChecker.TextProgressBar();
            this.SuspendLayout();
            // 
            // installButton
            // 
            this.installButton.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.installButton.Location = new System.Drawing.Point(12, 64);
            this.installButton.Name = "installButton";
            this.installButton.Size = new System.Drawing.Size(75, 23);
            this.installButton.TabIndex = 0;
            this.installButton.Text = "Install";
            this.installButton.UseVisualStyleBackColor = true;
            this.installButton.Click += new System.EventHandler(this.InstallButtonClick);
            // 
            // cancelButton
            // 
            this.cancelButton.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.cancelButton.DialogResult = System.Windows.Forms.DialogResult.Cancel;
            this.cancelButton.Location = new System.Drawing.Point(253, 64);
            this.cancelButton.Name = "cancelButton";
            this.cancelButton.Size = new System.Drawing.Size(75, 23);
            this.cancelButton.TabIndex = 1;
            this.cancelButton.Text = "Cancel";
            this.cancelButton.UseVisualStyleBackColor = true;
            this.cancelButton.Click += new System.EventHandler(this.CancelButtonClick);
            // 
            // label1
            // 
            this.label1.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(81, 26);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(178, 13);
            this.label1.TabIndex = 2;
            this.label1.Text = "A new update for MPDN is available";
            // 
            // forgetUpdateButton
            // 
            this.forgetUpdateButton.Anchor = System.Windows.Forms.AnchorStyles.Bottom;
            this.forgetUpdateButton.Location = new System.Drawing.Point(133, 64);
            this.forgetUpdateButton.Name = "forgetUpdateButton";
            this.forgetUpdateButton.Size = new System.Drawing.Size(75, 23);
            this.forgetUpdateButton.TabIndex = 3;
            this.forgetUpdateButton.Text = "Forget Update";
            this.forgetUpdateButton.UseVisualStyleBackColor = true;
            this.forgetUpdateButton.Click += new System.EventHandler(this.ForgetUpdateButtonClick);
            // 
            // downloadProgressBar
            // 
            this.downloadProgressBar.CustomText = null;
            this.downloadProgressBar.DisplayStyle = Mpdn.Extensions.PlayerExtensions.UpdateChecker.TextProgressBar.ProgressBarDisplayText.Both;
            this.downloadProgressBar.Dock = System.Windows.Forms.DockStyle.Top;
            this.downloadProgressBar.Location = new System.Drawing.Point(0, 0);
            this.downloadProgressBar.Name = "downloadProgressBar";
            this.downloadProgressBar.Size = new System.Drawing.Size(340, 23);
            this.downloadProgressBar.TabIndex = 4;
            this.downloadProgressBar.Visible = false;
            // 
            // SimpleUpdateForm
            // 
            this.AcceptButton = this.installButton;
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.CancelButton = this.cancelButton;
            this.ClientSize = new System.Drawing.Size(340, 99);
            this.ControlBox = false;
            this.Controls.Add(this.downloadProgressBar);
            this.Controls.Add(this.forgetUpdateButton);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.cancelButton);
            this.Controls.Add(this.installButton);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.Name = "SimpleUpdateForm";
            this.ShowInTaskbar = false;
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
            this.Text = "New Update Available";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button installButton;
        private System.Windows.Forms.Button cancelButton;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Button forgetUpdateButton;
        private TextProgressBar downloadProgressBar;
    }
}