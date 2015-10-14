namespace Mpdn.Extensions.PlayerExtensions.UpdateChecker
{
    sealed partial class ChangelogForm
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
            this.changeLogWebViewer = new Mpdn.Extensions.PlayerExtensions.UpdateChecker.ChangelogWebViewer();
            this.closeButton = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // changeLogWebViewer
            // 
            this.changeLogWebViewer.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.changeLogWebViewer.IsWebBrowserContextMenuEnabled = false;
            this.changeLogWebViewer.Location = new System.Drawing.Point(0, 0);
            this.changeLogWebViewer.MinimumSize = new System.Drawing.Size(20, 20);
            this.changeLogWebViewer.Name = "changeLogWebViewer";
            this.changeLogWebViewer.Size = new System.Drawing.Size(624, 304);
            this.changeLogWebViewer.TabIndex = 0;
            this.changeLogWebViewer.WebBrowserShortcutsEnabled = false;
            // 
            // closeButton
            // 
            this.closeButton.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.closeButton.Location = new System.Drawing.Point(537, 322);
            this.closeButton.Name = "closeButton";
            this.closeButton.Size = new System.Drawing.Size(75, 23);
            this.closeButton.TabIndex = 1;
            this.closeButton.Text = "Close";
            this.closeButton.UseVisualStyleBackColor = true;
            this.closeButton.Click += new System.EventHandler(this.CloseButtonClick);
            // 
            // ChangelogForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(624, 357);
            this.Controls.Add(this.closeButton);
            this.Controls.Add(this.changeLogWebViewer);
            this.Name = "ChangelogForm";
            this.Text = "SimpleUpdate";
            this.ResumeLayout(false);

        }

        #endregion

        private ChangelogWebViewer changeLogWebViewer;
        private System.Windows.Forms.Button closeButton;
    }
}