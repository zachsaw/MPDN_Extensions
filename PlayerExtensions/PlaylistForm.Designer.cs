namespace Mpdn.PlayerExtensions.GitHub
{
    partial class PlaylistForm
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
            this.components = new System.ComponentModel.Container();
            this.statusStrip1 = new System.Windows.Forms.StatusStrip();
            this.buttonAdd = new ButtonStripItem();
            this.buttonDel = new ButtonStripItem();
            this.toolStripStatusLabel2 = new System.Windows.Forms.ToolStripStatusLabel();
            this.buttonLeft = new ButtonStripItem();
            this.buttonRight = new ButtonStripItem();
            this.toolStripStatusLabel1 = new System.Windows.Forms.ToolStripStatusLabel();
            this.buttonOpen = new ButtonStripItem();
            this.buttonSave = new ButtonStripItem();
            this.listBox = new DragDropListBox();
            this.contextMenuStrip1 = new System.Windows.Forms.ContextMenuStrip(this.components);
            this.selectAllToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripMenuItem3 = new System.Windows.Forms.ToolStripSeparator();
            this.addFilesToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.removeFilesToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripMenuItem1 = new System.Windows.Forms.ToolStripSeparator();
            this.nextToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.previousToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripMenuItem2 = new System.Windows.Forms.ToolStripSeparator();
            this.openPlaylistToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.savePlaylistToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.timer = new System.Windows.Forms.Timer(this.components);
            this.openFileDialog = new System.Windows.Forms.OpenFileDialog();
            this.openPlaylistDialog = new System.Windows.Forms.OpenFileDialog();
            this.savePlaylistDialog = new System.Windows.Forms.SaveFileDialog();
            this.statusStrip1.SuspendLayout();
            this.contextMenuStrip1.SuspendLayout();
            this.SuspendLayout();
            // 
            // statusStrip1
            // 
            this.statusStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.buttonAdd,
            this.buttonDel,
            this.toolStripStatusLabel2,
            this.buttonLeft,
            this.buttonRight,
            this.toolStripStatusLabel1,
            this.buttonOpen,
            this.buttonSave});
            this.statusStrip1.Location = new System.Drawing.Point(0, 194);
            this.statusStrip1.Name = "statusStrip1";
            this.statusStrip1.RenderMode = System.Windows.Forms.ToolStripRenderMode.Professional;
            this.statusStrip1.ShowItemToolTips = true;
            this.statusStrip1.Size = new System.Drawing.Size(224, 27);
            this.statusStrip1.TabIndex = 0;
            this.statusStrip1.TabStop = true;
            this.statusStrip1.Text = "statusStrip1";
            // 
            // buttonAdd
            // 
            this.buttonAdd.AutoSize = false;
            this.buttonAdd.Name = "buttonAdd";
            this.buttonAdd.Size = new System.Drawing.Size(25, 25);
            this.buttonAdd.Text = "+";
            this.buttonAdd.ToolTipText = "Add files...";
            this.buttonAdd.Click += new System.EventHandler(this.ButtonAddClick);
            // 
            // buttonDel
            // 
            this.buttonDel.AutoSize = false;
            this.buttonDel.Name = "buttonDel";
            this.buttonDel.Size = new System.Drawing.Size(25, 25);
            this.buttonDel.Text = "-";
            this.buttonDel.ToolTipText = "Remove files";
            this.buttonDel.Click += new System.EventHandler(this.ButtonDelClick);
            // 
            // toolStripStatusLabel2
            // 
            this.toolStripStatusLabel2.Name = "toolStripStatusLabel2";
            this.toolStripStatusLabel2.Size = new System.Drawing.Size(10, 22);
            this.toolStripStatusLabel2.Text = " ";
            // 
            // buttonLeft
            // 
            this.buttonLeft.AutoSize = false;
            this.buttonLeft.Name = "buttonLeft";
            this.buttonLeft.Size = new System.Drawing.Size(25, 25);
            this.buttonLeft.Text = "<";
            this.buttonLeft.ToolTipText = "Previous";
            this.buttonLeft.Click += new System.EventHandler(this.ButtonLeftClick);
            // 
            // buttonRight
            // 
            this.buttonRight.AutoSize = false;
            this.buttonRight.Name = "buttonRight";
            this.buttonRight.Size = new System.Drawing.Size(25, 25);
            this.buttonRight.Text = ">";
            this.buttonRight.ToolTipText = "Next";
            this.buttonRight.Click += new System.EventHandler(this.ButtonRightClick);
            // 
            // toolStripStatusLabel1
            // 
            this.toolStripStatusLabel1.Name = "toolStripStatusLabel1";
            this.toolStripStatusLabel1.Size = new System.Drawing.Size(9, 22);
            this.toolStripStatusLabel1.Spring = true;
            this.toolStripStatusLabel1.Text = " ";
            // 
            // buttonOpen
            // 
            this.buttonOpen.AutoSize = false;
            this.buttonOpen.Name = "buttonOpen";
            this.buttonOpen.Size = new System.Drawing.Size(45, 25);
            this.buttonOpen.Text = "Open";
            this.buttonOpen.ToolTipText = "Open play list";
            this.buttonOpen.Click += new System.EventHandler(this.ButtonOpenClick);
            // 
            // buttonSave
            // 
            this.buttonSave.AutoSize = false;
            this.buttonSave.Name = "buttonSave";
            this.buttonSave.Size = new System.Drawing.Size(45, 25);
            this.buttonSave.Text = "Save";
            this.buttonSave.ToolTipText = "Save play list";
            this.buttonSave.Click += new System.EventHandler(this.ButtonSaveClick);
            // 
            // listBox
            // 
            this.listBox.AllowDrop = true;
            this.listBox.ContextMenuStrip = this.contextMenuStrip1;
            this.listBox.Dock = System.Windows.Forms.DockStyle.Fill;
            this.listBox.DrawMode = System.Windows.Forms.DrawMode.OwnerDrawFixed;
            this.listBox.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.listBox.FormattingEnabled = true;
            this.listBox.IntegralHeight = false;
            this.listBox.IsDragDropCopySource = false;
            this.listBox.IsDragDropMoveSource = false;
            this.listBox.ItemHeight = 15;
            this.listBox.Location = new System.Drawing.Point(0, 0);
            this.listBox.Name = "listBox";
            this.listBox.SelectionMode = System.Windows.Forms.SelectionMode.MultiExtended;
            this.listBox.Size = new System.Drawing.Size(224, 194);
            this.listBox.TabIndex = 0;
            this.listBox.Dropped += new System.EventHandler<DroppedEventArgs>(this.ListBoxDropped);
            this.listBox.DrawItem += new System.Windows.Forms.DrawItemEventHandler(this.ListBoxDrawItem);
            this.listBox.SizeChanged += new System.EventHandler(this.ListBoxSizeChanged);
            this.listBox.DragDrop += new System.Windows.Forms.DragEventHandler(this.ListBoxDragDrop);
            this.listBox.DragEnter += new System.Windows.Forms.DragEventHandler(this.ListBoxDragEnter);
            this.listBox.DoubleClick += new System.EventHandler(this.ListBoxDoubleClick);
            // 
            // contextMenuStrip1
            // 
            this.contextMenuStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.selectAllToolStripMenuItem,
            this.toolStripMenuItem3,
            this.addFilesToolStripMenuItem,
            this.removeFilesToolStripMenuItem,
            this.toolStripMenuItem1,
            this.nextToolStripMenuItem,
            this.previousToolStripMenuItem,
            this.toolStripMenuItem2,
            this.openPlaylistToolStripMenuItem,
            this.savePlaylistToolStripMenuItem});
            this.contextMenuStrip1.Name = "contextMenuStrip1";
            this.contextMenuStrip1.Size = new System.Drawing.Size(219, 176);
            // 
            // selectAllToolStripMenuItem
            // 
            this.selectAllToolStripMenuItem.Name = "selectAllToolStripMenuItem";
            this.selectAllToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.A)));
            this.selectAllToolStripMenuItem.Size = new System.Drawing.Size(218, 22);
            this.selectAllToolStripMenuItem.Text = "Select All";
            this.selectAllToolStripMenuItem.Click += new System.EventHandler(this.SelectAllClick);
            // 
            // toolStripMenuItem3
            // 
            this.toolStripMenuItem3.Name = "toolStripMenuItem3";
            this.toolStripMenuItem3.Size = new System.Drawing.Size(215, 6);
            // 
            // addFilesToolStripMenuItem
            // 
            this.addFilesToolStripMenuItem.Name = "addFilesToolStripMenuItem";
            this.addFilesToolStripMenuItem.ShortcutKeys = System.Windows.Forms.Keys.Insert;
            this.addFilesToolStripMenuItem.Size = new System.Drawing.Size(218, 22);
            this.addFilesToolStripMenuItem.Text = "Add files...";
            this.addFilesToolStripMenuItem.Click += new System.EventHandler(this.ButtonAddClick);
            // 
            // removeFilesToolStripMenuItem
            // 
            this.removeFilesToolStripMenuItem.Name = "removeFilesToolStripMenuItem";
            this.removeFilesToolStripMenuItem.ShortcutKeys = System.Windows.Forms.Keys.Delete;
            this.removeFilesToolStripMenuItem.Size = new System.Drawing.Size(218, 22);
            this.removeFilesToolStripMenuItem.Text = "Remove files";
            this.removeFilesToolStripMenuItem.Click += new System.EventHandler(this.ButtonDelClick);
            // 
            // toolStripMenuItem1
            // 
            this.toolStripMenuItem1.Name = "toolStripMenuItem1";
            this.toolStripMenuItem1.Size = new System.Drawing.Size(215, 6);
            // 
            // nextToolStripMenuItem
            // 
            this.nextToolStripMenuItem.Name = "nextToolStripMenuItem";
            this.nextToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)(((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.Alt) 
            | System.Windows.Forms.Keys.N)));
            this.nextToolStripMenuItem.Size = new System.Drawing.Size(218, 22);
            this.nextToolStripMenuItem.Text = "&Next";
            this.nextToolStripMenuItem.Click += new System.EventHandler(this.ButtonRightClick);
            // 
            // previousToolStripMenuItem
            // 
            this.previousToolStripMenuItem.Name = "previousToolStripMenuItem";
            this.previousToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)(((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.Alt) 
            | System.Windows.Forms.Keys.B)));
            this.previousToolStripMenuItem.Size = new System.Drawing.Size(218, 22);
            this.previousToolStripMenuItem.Text = "&Previous";
            this.previousToolStripMenuItem.Click += new System.EventHandler(this.ButtonLeftClick);
            // 
            // toolStripMenuItem2
            // 
            this.toolStripMenuItem2.Name = "toolStripMenuItem2";
            this.toolStripMenuItem2.Size = new System.Drawing.Size(215, 6);
            // 
            // openPlaylistToolStripMenuItem
            // 
            this.openPlaylistToolStripMenuItem.Name = "openPlaylistToolStripMenuItem";
            this.openPlaylistToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)(((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.Alt) 
            | System.Windows.Forms.Keys.O)));
            this.openPlaylistToolStripMenuItem.Size = new System.Drawing.Size(218, 22);
            this.openPlaylistToolStripMenuItem.Text = "&Open playlist...";
            this.openPlaylistToolStripMenuItem.Click += new System.EventHandler(this.ButtonOpenClick);
            // 
            // savePlaylistToolStripMenuItem
            // 
            this.savePlaylistToolStripMenuItem.Name = "savePlaylistToolStripMenuItem";
            this.savePlaylistToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)(((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.Alt) 
            | System.Windows.Forms.Keys.S)));
            this.savePlaylistToolStripMenuItem.Size = new System.Drawing.Size(218, 22);
            this.savePlaylistToolStripMenuItem.Text = "&Save playlist...";
            this.savePlaylistToolStripMenuItem.Click += new System.EventHandler(this.ButtonSaveClick);
            // 
            // timer
            // 
            this.timer.Interval = 15;
            this.timer.Tick += new System.EventHandler(this.TimerTick);
            // 
            // openFileDialog
            // 
            this.openFileDialog.Multiselect = true;
            this.openFileDialog.RestoreDirectory = true;
            this.openFileDialog.SupportMultiDottedExtensions = true;
            this.openFileDialog.Title = "Add file...";
            // 
            // openPlaylistDialog
            // 
            this.openPlaylistDialog.DefaultExt = "mpl";
            this.openPlaylistDialog.Filter = "MPDN Playlist (*.mpl) |*.mpl|All files (*.*)|*.*";
            this.openPlaylistDialog.RestoreDirectory = true;
            this.openPlaylistDialog.SupportMultiDottedExtensions = true;
            this.openPlaylistDialog.Title = "Open playlist...";
            // 
            // savePlaylistDialog
            // 
            this.savePlaylistDialog.DefaultExt = "mpl";
            this.savePlaylistDialog.Filter = "MPDN Playlist (*.mpl) |*.mpl|All files (*.*)|*.*";
            this.savePlaylistDialog.SupportMultiDottedExtensions = true;
            this.savePlaylistDialog.Title = "Save playlist...";
            // 
            // PlaylistForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(224, 221);
            this.Controls.Add(this.listBox);
            this.Controls.Add(this.statusStrip1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.SizableToolWindow;
            this.KeyPreview = true;
            this.MinimumSize = new System.Drawing.Size(240, 260);
            this.Name = "PlaylistForm";
            this.StartPosition = System.Windows.Forms.FormStartPosition.Manual;
            this.Text = "Play List";
            this.TopMost = true;
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.PlaylistFormClosing);
            this.KeyDown += new System.Windows.Forms.KeyEventHandler(this.FormKeyDown);
            this.statusStrip1.ResumeLayout(false);
            this.statusStrip1.PerformLayout();
            this.contextMenuStrip1.ResumeLayout(false);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.StatusStrip statusStrip1;
        private ButtonStripItem buttonAdd;
        private ButtonStripItem buttonDel;
        private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabel1;
        private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabel2;
        private ButtonStripItem buttonOpen;
        private ButtonStripItem buttonSave;
        private DragDropListBox listBox;
        private System.Windows.Forms.Timer timer;
        private System.Windows.Forms.OpenFileDialog openFileDialog;
        private System.Windows.Forms.ContextMenuStrip contextMenuStrip1;
        private System.Windows.Forms.ToolStripMenuItem removeFilesToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem addFilesToolStripMenuItem;
        private System.Windows.Forms.ToolStripSeparator toolStripMenuItem1;
        private System.Windows.Forms.ToolStripMenuItem openPlaylistToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem savePlaylistToolStripMenuItem;
        private ButtonStripItem buttonLeft;
        private ButtonStripItem buttonRight;
        private System.Windows.Forms.ToolStripMenuItem nextToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem previousToolStripMenuItem;
        private System.Windows.Forms.ToolStripSeparator toolStripMenuItem2;
        private System.Windows.Forms.OpenFileDialog openPlaylistDialog;
        private System.Windows.Forms.SaveFileDialog savePlaylistDialog;
        private System.Windows.Forms.ToolStripMenuItem selectAllToolStripMenuItem;
        private System.Windows.Forms.ToolStripSeparator toolStripMenuItem3;
    }
}