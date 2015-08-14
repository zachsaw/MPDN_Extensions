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
            this.playerLabel = new System.Windows.Forms.Label();
            this.forgetUpdateButton = new System.Windows.Forms.Button();
            this.downloadProgressBar = new Mpdn.Extensions.PlayerExtensions.UpdateChecker.TextProgressBar();
            this.extensionLabel = new System.Windows.Forms.Label();
            this.playerVersionLinkLabel = new System.Windows.Forms.LinkLabel();
            this.extensionVersionLinkLabel = new System.Windows.Forms.LinkLabel();
            this.SuspendLayout();
            // 
            // installButton
            // 
            this.installButton.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.installButton.Location = new System.Drawing.Point(12, 92);
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
            this.cancelButton.Location = new System.Drawing.Point(287, 92);
            this.cancelButton.Name = "cancelButton";
            this.cancelButton.Size = new System.Drawing.Size(75, 23);
            this.cancelButton.TabIndex = 1;
            this.cancelButton.Text = "Cancel";
            this.cancelButton.UseVisualStyleBackColor = true;
            this.cancelButton.Click += new System.EventHandler(this.CancelButtonClick);
            // 
            // playerLabel
            // 
            this.playerLabel.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.playerLabel.AutoSize = true;
            this.playerLabel.Location = new System.Drawing.Point(81, 26);
            this.playerLabel.Name = "playerLabel";
            this.playerLabel.Size = new System.Drawing.Size(112, 13);
            this.playerLabel.TabIndex = 2;
            this.playerLabel.Text = "New Player available: ";
            this.playerLabel.Visible = false;
            // 
            // forgetUpdateButton
            // 
            this.forgetUpdateButton.Anchor = System.Windows.Forms.AnchorStyles.Bottom;
            this.forgetUpdateButton.Location = new System.Drawing.Point(150, 92);
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
            this.downloadProgressBar.Size = new System.Drawing.Size(374, 23);
            this.downloadProgressBar.TabIndex = 4;
            this.downloadProgressBar.Visible = false;
            // 
            // extensionLabel
            // 
            this.extensionLabel.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.extensionLabel.AutoSize = true;
            this.extensionLabel.Location = new System.Drawing.Point(81, 43);
            this.extensionLabel.Name = "extensionLabel";
            this.extensionLabel.Size = new System.Drawing.Size(134, 13);
            this.extensionLabel.TabIndex = 5;
            this.extensionLabel.Text = "New Extensions available: ";
            this.extensionLabel.Visible = false;
            // 
            // playerVersionLinkLabel
            // 
            this.playerVersionLinkLabel.AutoSize = true;
            this.playerVersionLinkLabel.Location = new System.Drawing.Point(217, 26);
            this.playerVersionLinkLabel.Name = "playerVersionLinkLabel";
            this.playerVersionLinkLabel.Size = new System.Drawing.Size(48, 13);
            this.playerVersionLinkLabel.TabIndex = 6;
            this.playerVersionLinkLabel.TabStop = true;
            this.playerVersionLinkLabel.Text = "pVersion";
            this.playerVersionLinkLabel.Visible = false;
            // 
            // extensionVersionLinkLabel
            // 
            this.extensionVersionLinkLabel.AutoSize = true;
            this.extensionVersionLinkLabel.Location = new System.Drawing.Point(217, 43);
            this.extensionVersionLinkLabel.Name = "extensionVersionLinkLabel";
            this.extensionVersionLinkLabel.Size = new System.Drawing.Size(48, 13);
            this.extensionVersionLinkLabel.TabIndex = 7;
            this.extensionVersionLinkLabel.TabStop = true;
            this.extensionVersionLinkLabel.Text = "eVersion";
            this.extensionVersionLinkLabel.Visible = false;
            // 
            // SimpleUpdateForm
            // 
            this.AcceptButton = this.installButton;
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.CancelButton = this.cancelButton;
            this.ClientSize = new System.Drawing.Size(374, 127);
            this.ControlBox = false;
            this.Controls.Add(this.extensionVersionLinkLabel);
            this.Controls.Add(this.playerVersionLinkLabel);
            this.Controls.Add(this.extensionLabel);
            this.Controls.Add(this.downloadProgressBar);
            this.Controls.Add(this.forgetUpdateButton);
            this.Controls.Add(this.playerLabel);
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
        private System.Windows.Forms.Label playerLabel;
        private System.Windows.Forms.Button forgetUpdateButton;
        private TextProgressBar downloadProgressBar;
        private System.Windows.Forms.Label extensionLabel;
        private System.Windows.Forms.LinkLabel playerVersionLinkLabel;
        private System.Windows.Forms.LinkLabel extensionVersionLinkLabel;
    }
}