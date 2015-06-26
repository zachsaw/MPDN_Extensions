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
using System.Windows.Forms;

namespace Mpdn.Extensions.PlayerExtensions.Playlist
{
    partial class PlaylistForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(PlaylistForm));
            System.Windows.Forms.DataGridViewCellStyle dataGridViewCellStyle1 = new System.Windows.Forms.DataGridViewCellStyle();
            System.Windows.Forms.DataGridViewCellStyle dataGridViewCellStyle5 = new System.Windows.Forms.DataGridViewCellStyle();
            System.Windows.Forms.DataGridViewCellStyle dataGridViewCellStyle2 = new System.Windows.Forms.DataGridViewCellStyle();
            System.Windows.Forms.DataGridViewCellStyle dataGridViewCellStyle3 = new System.Windows.Forms.DataGridViewCellStyle();
            System.Windows.Forms.DataGridViewCellStyle dataGridViewCellStyle4 = new System.Windows.Forms.DataGridViewCellStyle();
            this.statusStrip1 = new System.Windows.Forms.StatusStrip();
            this.buttonAdd = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonDel = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonAddFolder = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.toolStripStatusLabel2 = new System.Windows.Forms.ToolStripStatusLabel();
            this.buttonLeft = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonRight = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonSortAscending = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonSortDescending = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonShuffle = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonRestore = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.PlayButton = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.PauseButton = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.StopButton = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonSaveDisabled = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonNewDisabled = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonNewEnabled = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonSaveEnabled = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonLeftDisabled = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonRightDisabled = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonLeftEnabled = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.toolStripStatusLabel1 = new System.Windows.Forms.ToolStripStatusLabel();
            this.buttonRightEnabled = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonDelDisabled = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonDelEnabled = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonShuffleEnabled = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonShuffleDisabled = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonRestoreEnabled = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonRestoreDisabled = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonSortDescendingDisabled = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonSortDescendingEnabled = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonSortAscendingEnabled = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonSortAscendingDisabled = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonNew = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonOpen = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonSave = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonSettings = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.timer = new System.Windows.Forms.Timer(this.components);
            this.openFileDialog = new System.Windows.Forms.OpenFileDialog();
            this.openPlaylistDialog = new System.Windows.Forms.OpenFileDialog();
            this.savePlaylistDialog = new System.Windows.Forms.SaveFileDialog();
            this.dgv_PlayList = new System.Windows.Forms.DataGridView();
            this.Playing = new System.Windows.Forms.DataGridViewImageColumn();
            this.Number = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.FullPath = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.CurrentDirectory = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.Title = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.SkipChapters = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.EndChapter = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.dgv_PlaylistContextMenu = new System.Windows.Forms.ContextMenuStrip(this.components);
            this.playFileToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.addFilesToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.addFromClipboardToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.removeFilesToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripMenuItem4 = new System.Windows.Forms.ToolStripSeparator();
            this.addFolderToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.openFolderToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripMenuItem1 = new System.Windows.Forms.ToolStripSeparator();
            this.openPlaylistToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.newPlaylistToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.savePlaylistToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripMenuItem2 = new System.Windows.Forms.ToolStripSeparator();
            this.nextItemToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.previousItemToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripMenuItem3 = new System.Windows.Forms.ToolStripSeparator();
            this.sortToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.ascendingToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.descendingToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.restoreToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.shuffleToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.dgv_PlaylistColumnContextMenu = new System.Windows.Forms.ContextMenuStrip(this.components);
            this.numberToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.fullPathToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.directoryToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.skipChaptersToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.endChapterToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.statusStrip1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.dgv_PlayList)).BeginInit();
            this.dgv_PlaylistContextMenu.SuspendLayout();
            this.dgv_PlaylistColumnContextMenu.SuspendLayout();
            this.SuspendLayout();
            // 
            // statusStrip1
            // 
            this.statusStrip1.BackColor = System.Drawing.SystemColors.Window;
            this.statusStrip1.ImageScalingSize = new System.Drawing.Size(20, 20);
            this.statusStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.buttonAdd,
            this.buttonDel,
            this.buttonAddFolder,
            this.toolStripStatusLabel2,
            this.buttonLeft,
            this.buttonRight,
            this.buttonSortAscending,
            this.buttonSortDescending,
            this.buttonShuffle,
            this.buttonRestore,
            this.PlayButton,
            this.PauseButton,
            this.StopButton,
            this.buttonSaveDisabled,
            this.buttonNewDisabled,
            this.buttonNewEnabled,
            this.buttonSaveEnabled,
            this.buttonLeftDisabled,
            this.buttonRightDisabled,
            this.buttonLeftEnabled,
            this.toolStripStatusLabel1,
            this.buttonRightEnabled,
            this.buttonDelDisabled,
            this.buttonDelEnabled,
            this.buttonShuffleEnabled,
            this.buttonShuffleDisabled,
            this.buttonRestoreEnabled,
            this.buttonRestoreDisabled,
            this.buttonSortDescendingDisabled,
            this.buttonSortDescendingEnabled,
            this.buttonSortAscendingEnabled,
            this.buttonSortAscendingDisabled,
            this.buttonNew,
            this.buttonOpen,
            this.buttonSave,
            this.buttonSettings});
            this.statusStrip1.Location = new System.Drawing.Point(0, 206);
            this.statusStrip1.Name = "statusStrip1";
            this.statusStrip1.RenderMode = System.Windows.Forms.ToolStripRenderMode.ManagerRenderMode;
            this.statusStrip1.ShowItemToolTips = true;
            this.statusStrip1.Size = new System.Drawing.Size(795, 27);
            this.statusStrip1.TabIndex = 0;
            this.statusStrip1.TabStop = true;
            this.statusStrip1.Text = "statusStrip1";
            // 
            // buttonAdd
            // 
            this.buttonAdd.AutoSize = false;
            this.buttonAdd.BackColor = System.Drawing.Color.Transparent;
            this.buttonAdd.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("buttonAdd.BackgroundImage")));
            this.buttonAdd.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonAdd.Name = "buttonAdd";
            this.buttonAdd.Size = new System.Drawing.Size(24, 25);
            this.buttonAdd.ToolTipText = "Add file(s)";
            this.buttonAdd.Click += new System.EventHandler(this.ButtonAddClick);
            // 
            // buttonDel
            // 
            this.buttonDel.AutoSize = false;
            this.buttonDel.BackColor = System.Drawing.Color.Transparent;
            this.buttonDel.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("buttonDel.BackgroundImage")));
            this.buttonDel.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonDel.Name = "buttonDel";
            this.buttonDel.Size = new System.Drawing.Size(24, 25);
            this.buttonDel.ToolTipText = "Remove file(s)";
            this.buttonDel.Click += new System.EventHandler(this.ButtonDelClick);
            // 
            // buttonAddFolder
            // 
            this.buttonAddFolder.AutoSize = false;
            this.buttonAddFolder.BackColor = System.Drawing.Color.Transparent;
            this.buttonAddFolder.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("buttonAddFolder.BackgroundImage")));
            this.buttonAddFolder.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonAddFolder.Name = "buttonAddFolder";
            this.buttonAddFolder.Size = new System.Drawing.Size(24, 25);
            this.buttonAddFolder.ToolTipText = "Add folder";
            this.buttonAddFolder.Click += new System.EventHandler(this.ButtonAddFolderClick);
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
            this.buttonLeft.BackColor = System.Drawing.Color.Transparent;
            this.buttonLeft.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("buttonLeft.BackgroundImage")));
            this.buttonLeft.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonLeft.Name = "buttonLeft";
            this.buttonLeft.Size = new System.Drawing.Size(24, 25);
            this.buttonLeft.ToolTipText = "Previous";
            this.buttonLeft.Click += new System.EventHandler(this.ButtonLeftClick);
            // 
            // buttonRight
            // 
            this.buttonRight.AutoSize = false;
            this.buttonRight.BackColor = System.Drawing.Color.Transparent;
            this.buttonRight.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("buttonRight.BackgroundImage")));
            this.buttonRight.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonRight.Name = "buttonRight";
            this.buttonRight.Size = new System.Drawing.Size(24, 25);
            this.buttonRight.ToolTipText = "Next";
            this.buttonRight.Click += new System.EventHandler(this.ButtonRightClick);
            // 
            // buttonSortAscending
            // 
            this.buttonSortAscending.BackColor = System.Drawing.Color.Transparent;
            this.buttonSortAscending.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("buttonSortAscending.BackgroundImage")));
            this.buttonSortAscending.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonSortAscending.Margin = new System.Windows.Forms.Padding(10, 2, 0, 0);
            this.buttonSortAscending.Name = "buttonSortAscending";
            this.buttonSortAscending.Size = new System.Drawing.Size(24, 25);
            this.buttonSortAscending.ToolTipText = "Sort playlist (ascending)";
            this.buttonSortAscending.Click += new System.EventHandler(this.ButtonSortAscendingClick);
            // 
            // buttonSortDescending
            // 
            this.buttonSortDescending.BackColor = System.Drawing.Color.Transparent;
            this.buttonSortDescending.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("buttonSortDescending.BackgroundImage")));
            this.buttonSortDescending.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonSortDescending.Name = "buttonSortDescending";
            this.buttonSortDescending.Size = new System.Drawing.Size(24, 25);
            this.buttonSortDescending.ToolTipText = "Sort playlist (descending)";
            this.buttonSortDescending.Click += new System.EventHandler(this.ButtonSortDescendingClick);
            // 
            // buttonShuffle
            // 
            this.buttonShuffle.BackColor = System.Drawing.Color.Transparent;
            this.buttonShuffle.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("buttonShuffle.BackgroundImage")));
            this.buttonShuffle.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonShuffle.Name = "buttonShuffle";
            this.buttonShuffle.Size = new System.Drawing.Size(24, 25);
            this.buttonShuffle.ToolTipText = "Shuffle playlist";
            this.buttonShuffle.Click += new System.EventHandler(this.ButtonShuffleClick);
            // 
            // buttonRestore
            // 
            this.buttonRestore.AutoSize = false;
            this.buttonRestore.BackColor = System.Drawing.Color.Transparent;
            this.buttonRestore.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("buttonRestore.BackgroundImage")));
            this.buttonRestore.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonRestore.Name = "buttonRestore";
            this.buttonRestore.Size = new System.Drawing.Size(24, 25);
            this.buttonRestore.ToolTipText = "Restore playlist";
            this.buttonRestore.Click += new System.EventHandler(this.ButtonRestoreClick);
            // 
            // PlayButton
            // 
            this.PlayButton.BackColor = System.Drawing.Color.Transparent;
            this.PlayButton.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("PlayButton.BackgroundImage")));
            this.PlayButton.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.PlayButton.Name = "PlayButton";
            this.PlayButton.Size = new System.Drawing.Size(24, 25);
            this.PlayButton.Visible = false;
            // 
            // PauseButton
            // 
            this.PauseButton.BackColor = System.Drawing.Color.Transparent;
            this.PauseButton.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("PauseButton.BackgroundImage")));
            this.PauseButton.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.PauseButton.Name = "PauseButton";
            this.PauseButton.Size = new System.Drawing.Size(24, 25);
            this.PauseButton.Visible = false;
            // 
            // StopButton
            // 
            this.StopButton.BackColor = System.Drawing.Color.Transparent;
            this.StopButton.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("StopButton.BackgroundImage")));
            this.StopButton.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.StopButton.Name = "StopButton";
            this.StopButton.Size = new System.Drawing.Size(24, 25);
            this.StopButton.Visible = false;
            // 
            // buttonSaveDisabled
            // 
            this.buttonSaveDisabled.BackColor = System.Drawing.Color.Transparent;
            this.buttonSaveDisabled.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("buttonSaveDisabled.BackgroundImage")));
            this.buttonSaveDisabled.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonSaveDisabled.Name = "buttonSaveDisabled";
            this.buttonSaveDisabled.Size = new System.Drawing.Size(24, 25);
            this.buttonSaveDisabled.Visible = false;
            // 
            // buttonNewDisabled
            // 
            this.buttonNewDisabled.BackColor = System.Drawing.Color.Transparent;
            this.buttonNewDisabled.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("buttonNewDisabled.BackgroundImage")));
            this.buttonNewDisabled.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonNewDisabled.Name = "buttonNewDisabled";
            this.buttonNewDisabled.Size = new System.Drawing.Size(24, 25);
            this.buttonNewDisabled.Visible = false;
            // 
            // buttonNewEnabled
            // 
            this.buttonNewEnabled.AutoSize = false;
            this.buttonNewEnabled.BackColor = System.Drawing.Color.Transparent;
            this.buttonNewEnabled.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("buttonNewEnabled.BackgroundImage")));
            this.buttonNewEnabled.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonNewEnabled.Name = "buttonNewEnabled";
            this.buttonNewEnabled.Size = new System.Drawing.Size(24, 25);
            this.buttonNewEnabled.ToolTipText = "New playlist";
            this.buttonNewEnabled.Visible = false;
            // 
            // buttonSaveEnabled
            // 
            this.buttonSaveEnabled.AutoSize = false;
            this.buttonSaveEnabled.BackColor = System.Drawing.Color.Transparent;
            this.buttonSaveEnabled.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("buttonSaveEnabled.BackgroundImage")));
            this.buttonSaveEnabled.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonSaveEnabled.Name = "buttonSaveEnabled";
            this.buttonSaveEnabled.Size = new System.Drawing.Size(24, 25);
            this.buttonSaveEnabled.ToolTipText = "Save playlist";
            this.buttonSaveEnabled.Visible = false;
            // 
            // buttonLeftDisabled
            // 
            this.buttonLeftDisabled.BackColor = System.Drawing.Color.Transparent;
            this.buttonLeftDisabled.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("buttonLeftDisabled.BackgroundImage")));
            this.buttonLeftDisabled.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonLeftDisabled.Name = "buttonLeftDisabled";
            this.buttonLeftDisabled.Size = new System.Drawing.Size(24, 25);
            this.buttonLeftDisabled.Visible = false;
            // 
            // buttonRightDisabled
            // 
            this.buttonRightDisabled.BackColor = System.Drawing.Color.Transparent;
            this.buttonRightDisabled.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("buttonRightDisabled.BackgroundImage")));
            this.buttonRightDisabled.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonRightDisabled.Name = "buttonRightDisabled";
            this.buttonRightDisabled.Size = new System.Drawing.Size(24, 25);
            this.buttonRightDisabled.Visible = false;
            // 
            // buttonLeftEnabled
            // 
            this.buttonLeftEnabled.AutoSize = false;
            this.buttonLeftEnabled.BackColor = System.Drawing.Color.Transparent;
            this.buttonLeftEnabled.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("buttonLeftEnabled.BackgroundImage")));
            this.buttonLeftEnabled.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonLeftEnabled.Name = "buttonLeftEnabled";
            this.buttonLeftEnabled.Size = new System.Drawing.Size(24, 25);
            this.buttonLeftEnabled.ToolTipText = "Previous";
            this.buttonLeftEnabled.Visible = false;
            // 
            // toolStripStatusLabel1
            // 
            this.toolStripStatusLabel1.Name = "toolStripStatusLabel1";
            this.toolStripStatusLabel1.Size = new System.Drawing.Size(448, 22);
            this.toolStripStatusLabel1.Spring = true;
            // 
            // buttonRightEnabled
            // 
            this.buttonRightEnabled.AutoSize = false;
            this.buttonRightEnabled.BackColor = System.Drawing.Color.Transparent;
            this.buttonRightEnabled.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("buttonRightEnabled.BackgroundImage")));
            this.buttonRightEnabled.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonRightEnabled.Name = "buttonRightEnabled";
            this.buttonRightEnabled.Size = new System.Drawing.Size(24, 25);
            this.buttonRightEnabled.ToolTipText = "Next";
            this.buttonRightEnabled.Visible = false;
            // 
            // buttonDelDisabled
            // 
            this.buttonDelDisabled.BackColor = System.Drawing.Color.Transparent;
            this.buttonDelDisabled.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("buttonDelDisabled.BackgroundImage")));
            this.buttonDelDisabled.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonDelDisabled.Name = "buttonDelDisabled";
            this.buttonDelDisabled.Size = new System.Drawing.Size(24, 25);
            this.buttonDelDisabled.Visible = false;
            // 
            // buttonDelEnabled
            // 
            this.buttonDelEnabled.AutoSize = false;
            this.buttonDelEnabled.BackColor = System.Drawing.Color.Transparent;
            this.buttonDelEnabled.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("buttonDelEnabled.BackgroundImage")));
            this.buttonDelEnabled.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonDelEnabled.Name = "buttonDelEnabled";
            this.buttonDelEnabled.Size = new System.Drawing.Size(24, 25);
            this.buttonDelEnabled.ToolTipText = "Remove file(s)";
            this.buttonDelEnabled.Visible = false;
            // 
            // buttonShuffleEnabled
            // 
            this.buttonShuffleEnabled.AutoSize = false;
            this.buttonShuffleEnabled.BackColor = System.Drawing.Color.Transparent;
            this.buttonShuffleEnabled.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("buttonShuffleEnabled.BackgroundImage")));
            this.buttonShuffleEnabled.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonShuffleEnabled.Name = "buttonShuffleEnabled";
            this.buttonShuffleEnabled.Size = new System.Drawing.Size(24, 25);
            this.buttonShuffleEnabled.ToolTipText = "Shuffle playlist";
            this.buttonShuffleEnabled.Visible = false;
            // 
            // buttonShuffleDisabled
            // 
            this.buttonShuffleDisabled.AutoSize = false;
            this.buttonShuffleDisabled.BackColor = System.Drawing.Color.Transparent;
            this.buttonShuffleDisabled.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("buttonShuffleDisabled.BackgroundImage")));
            this.buttonShuffleDisabled.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonShuffleDisabled.Name = "buttonShuffleDisabled";
            this.buttonShuffleDisabled.Size = new System.Drawing.Size(24, 25);
            this.buttonShuffleDisabled.ToolTipText = "Shuffle playlist";
            this.buttonShuffleDisabled.Visible = false;
            // 
            // buttonRestoreEnabled
            // 
            this.buttonRestoreEnabled.AutoSize = false;
            this.buttonRestoreEnabled.BackColor = System.Drawing.Color.Transparent;
            this.buttonRestoreEnabled.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("buttonRestoreEnabled.BackgroundImage")));
            this.buttonRestoreEnabled.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonRestoreEnabled.Name = "buttonRestoreEnabled";
            this.buttonRestoreEnabled.Size = new System.Drawing.Size(24, 25);
            this.buttonRestoreEnabled.ToolTipText = "Restore playlist";
            this.buttonRestoreEnabled.Visible = false;
            // 
            // buttonRestoreDisabled
            // 
            this.buttonRestoreDisabled.AutoSize = false;
            this.buttonRestoreDisabled.BackColor = System.Drawing.Color.Transparent;
            this.buttonRestoreDisabled.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("buttonRestoreDisabled.BackgroundImage")));
            this.buttonRestoreDisabled.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonRestoreDisabled.Name = "buttonRestoreDisabled";
            this.buttonRestoreDisabled.Size = new System.Drawing.Size(24, 25);
            this.buttonRestoreDisabled.ToolTipText = "Restore playlist";
            this.buttonRestoreDisabled.Visible = false;
            // 
            // buttonSortDescendingDisabled
            // 
            this.buttonSortDescendingDisabled.AutoSize = false;
            this.buttonSortDescendingDisabled.BackColor = System.Drawing.Color.Transparent;
            this.buttonSortDescendingDisabled.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("buttonSortDescendingDisabled.BackgroundImage")));
            this.buttonSortDescendingDisabled.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonSortDescendingDisabled.Name = "buttonSortDescendingDisabled";
            this.buttonSortDescendingDisabled.Size = new System.Drawing.Size(24, 25);
            this.buttonSortDescendingDisabled.ToolTipText = "Shuffle playlist";
            this.buttonSortDescendingDisabled.Visible = false;
            // 
            // buttonSortDescendingEnabled
            // 
            this.buttonSortDescendingEnabled.AutoSize = false;
            this.buttonSortDescendingEnabled.BackColor = System.Drawing.Color.Transparent;
            this.buttonSortDescendingEnabled.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("buttonSortDescendingEnabled.BackgroundImage")));
            this.buttonSortDescendingEnabled.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonSortDescendingEnabled.Name = "buttonSortDescendingEnabled";
            this.buttonSortDescendingEnabled.Size = new System.Drawing.Size(24, 25);
            this.buttonSortDescendingEnabled.ToolTipText = "Shuffle playlist";
            this.buttonSortDescendingEnabled.Visible = false;
            // 
            // buttonSortAscendingEnabled
            // 
            this.buttonSortAscendingEnabled.AutoSize = false;
            this.buttonSortAscendingEnabled.BackColor = System.Drawing.Color.Transparent;
            this.buttonSortAscendingEnabled.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("buttonSortAscendingEnabled.BackgroundImage")));
            this.buttonSortAscendingEnabled.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonSortAscendingEnabled.Name = "buttonSortAscendingEnabled";
            this.buttonSortAscendingEnabled.Size = new System.Drawing.Size(24, 25);
            this.buttonSortAscendingEnabled.ToolTipText = "Shuffle playlist";
            this.buttonSortAscendingEnabled.Visible = false;
            // 
            // buttonSortAscendingDisabled
            // 
            this.buttonSortAscendingDisabled.AutoSize = false;
            this.buttonSortAscendingDisabled.BackColor = System.Drawing.Color.Transparent;
            this.buttonSortAscendingDisabled.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("buttonSortAscendingDisabled.BackgroundImage")));
            this.buttonSortAscendingDisabled.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonSortAscendingDisabled.Name = "buttonSortAscendingDisabled";
            this.buttonSortAscendingDisabled.Size = new System.Drawing.Size(24, 25);
            this.buttonSortAscendingDisabled.ToolTipText = "Shuffle playlist";
            this.buttonSortAscendingDisabled.Visible = false;
            // 
            // buttonNew
            // 
            this.buttonNew.AutoSize = false;
            this.buttonNew.BackColor = System.Drawing.Color.Transparent;
            this.buttonNew.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("buttonNew.BackgroundImage")));
            this.buttonNew.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonNew.Name = "buttonNew";
            this.buttonNew.Size = new System.Drawing.Size(24, 25);
            this.buttonNew.ToolTipText = "New playlist";
            this.buttonNew.Click += new System.EventHandler(this.ButtonNewClick);
            // 
            // buttonOpen
            // 
            this.buttonOpen.AutoSize = false;
            this.buttonOpen.BackColor = System.Drawing.Color.Transparent;
            this.buttonOpen.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("buttonOpen.BackgroundImage")));
            this.buttonOpen.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonOpen.Name = "buttonOpen";
            this.buttonOpen.Size = new System.Drawing.Size(24, 25);
            this.buttonOpen.ToolTipText = "Open playlist";
            this.buttonOpen.Click += new System.EventHandler(this.ButtonOpenClick);
            // 
            // buttonSave
            // 
            this.buttonSave.AutoSize = false;
            this.buttonSave.BackColor = System.Drawing.Color.Transparent;
            this.buttonSave.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("buttonSave.BackgroundImage")));
            this.buttonSave.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonSave.Name = "buttonSave";
            this.buttonSave.Size = new System.Drawing.Size(24, 25);
            this.buttonSave.ToolTipText = "Save playlist";
            this.buttonSave.Click += new System.EventHandler(this.ButtonSaveClick);
            // 
            // buttonSettings
            // 
            this.buttonSettings.BackColor = System.Drawing.Color.Transparent;
            this.buttonSettings.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("buttonSettings.BackgroundImage")));
            this.buttonSettings.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonSettings.Name = "buttonSettings";
            this.buttonSettings.Size = new System.Drawing.Size(24, 25);
            this.buttonSettings.ToolTipText = "Configure playlist";
            this.buttonSettings.Click += new System.EventHandler(this.ButtonSettingsClick);
            // 
            // timer
            // 
            this.timer.Interval = 30;
            this.timer.Tick += new System.EventHandler(this.TimerTick);
            // 
            // openFileDialog
            // 
            this.openFileDialog.Filter = resources.GetString("openFileDialog.Filter");
            this.openFileDialog.Multiselect = true;
            this.openFileDialog.RestoreDirectory = true;
            this.openFileDialog.SupportMultiDottedExtensions = true;
            this.openFileDialog.Title = "Add file(s)...";
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
            // dgv_PlayList
            // 
            this.dgv_PlayList.AllowDrop = true;
            this.dgv_PlayList.AllowUserToAddRows = false;
            this.dgv_PlayList.AllowUserToDeleteRows = false;
            this.dgv_PlayList.AllowUserToResizeRows = false;
            dataGridViewCellStyle1.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(248)))), ((int)(((byte)(248)))), ((int)(((byte)(248)))));
            this.dgv_PlayList.AlternatingRowsDefaultCellStyle = dataGridViewCellStyle1;
            this.dgv_PlayList.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.dgv_PlayList.BackgroundColor = System.Drawing.SystemColors.Window;
            this.dgv_PlayList.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.dgv_PlayList.CellBorderStyle = System.Windows.Forms.DataGridViewCellBorderStyle.None;
            this.dgv_PlayList.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.dgv_PlayList.Columns.AddRange(new System.Windows.Forms.DataGridViewColumn[] {
            this.Playing,
            this.Number,
            this.FullPath,
            this.CurrentDirectory,
            this.Title,
            this.SkipChapters,
            this.EndChapter});
            dataGridViewCellStyle5.Alignment = System.Windows.Forms.DataGridViewContentAlignment.MiddleLeft;
            dataGridViewCellStyle5.BackColor = System.Drawing.Color.White;
            dataGridViewCellStyle5.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            dataGridViewCellStyle5.ForeColor = System.Drawing.Color.Black;
            dataGridViewCellStyle5.SelectionBackColor = System.Drawing.Color.SkyBlue;
            dataGridViewCellStyle5.SelectionForeColor = System.Drawing.Color.White;
            dataGridViewCellStyle5.WrapMode = System.Windows.Forms.DataGridViewTriState.False;
            this.dgv_PlayList.DefaultCellStyle = dataGridViewCellStyle5;
            this.dgv_PlayList.EditMode = System.Windows.Forms.DataGridViewEditMode.EditOnEnter;
            this.dgv_PlayList.GridColor = System.Drawing.SystemColors.Window;
            this.dgv_PlayList.Location = new System.Drawing.Point(0, 0);
            this.dgv_PlayList.Name = "dgv_PlayList";
            this.dgv_PlayList.RowHeadersVisible = false;
            this.dgv_PlayList.RowTemplate.Height = 29;
            this.dgv_PlayList.RowTemplate.Resizable = System.Windows.Forms.DataGridViewTriState.False;
            this.dgv_PlayList.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
            this.dgv_PlayList.SelectionMode = System.Windows.Forms.DataGridViewSelectionMode.FullRowSelect;
            this.dgv_PlayList.ShowCellErrors = false;
            this.dgv_PlayList.ShowCellToolTips = false;
            this.dgv_PlayList.ShowEditingIcon = false;
            this.dgv_PlayList.ShowRowErrors = false;
            this.dgv_PlayList.Size = new System.Drawing.Size(795, 206);
            this.dgv_PlayList.TabIndex = 1;
            this.dgv_PlayList.TabStop = false;
            // 
            // Playing
            // 
            this.Playing.AutoSizeMode = System.Windows.Forms.DataGridViewAutoSizeColumnMode.None;
            dataGridViewCellStyle2.Alignment = System.Windows.Forms.DataGridViewContentAlignment.MiddleCenter;
            dataGridViewCellStyle2.NullValue = "null";
            this.Playing.DefaultCellStyle = dataGridViewCellStyle2;
            this.Playing.FillWeight = 1F;
            this.Playing.HeaderText = "";
            this.Playing.MinimumWidth = 24;
            this.Playing.Name = "Playing";
            this.Playing.ReadOnly = true;
            this.Playing.Resizable = System.Windows.Forms.DataGridViewTriState.False;
            this.Playing.Width = 24;
            // 
            // Number
            // 
            this.Number.AutoSizeMode = System.Windows.Forms.DataGridViewAutoSizeColumnMode.None;
            this.Number.FillWeight = 1F;
            this.Number.HeaderText = "#";
            this.Number.MinimumWidth = 30;
            this.Number.Name = "Number";
            this.Number.ReadOnly = true;
            this.Number.SortMode = System.Windows.Forms.DataGridViewColumnSortMode.NotSortable;
            this.Number.Width = 30;
            // 
            // FullPath
            // 
            this.FullPath.AutoSizeMode = System.Windows.Forms.DataGridViewAutoSizeColumnMode.None;
            this.FullPath.FillWeight = 40F;
            this.FullPath.HeaderText = "Full Path";
            this.FullPath.MinimumWidth = 60;
            this.FullPath.Name = "FullPath";
            this.FullPath.ReadOnly = true;
            this.FullPath.SortMode = System.Windows.Forms.DataGridViewColumnSortMode.NotSortable;
            this.FullPath.Width = 150;
            // 
            // CurrentDirectory
            // 
            this.CurrentDirectory.AutoSizeMode = System.Windows.Forms.DataGridViewAutoSizeColumnMode.None;
            this.CurrentDirectory.FillWeight = 45F;
            this.CurrentDirectory.HeaderText = "Directory";
            this.CurrentDirectory.MinimumWidth = 60;
            this.CurrentDirectory.Name = "CurrentDirectory";
            this.CurrentDirectory.ReadOnly = true;
            this.CurrentDirectory.SortMode = System.Windows.Forms.DataGridViewColumnSortMode.NotSortable;
            this.CurrentDirectory.Visible = false;
            // 
            // Title
            // 
            this.Title.AutoSizeMode = System.Windows.Forms.DataGridViewAutoSizeColumnMode.None;
            this.Title.FillWeight = 50F;
            this.Title.HeaderText = "Title";
            this.Title.MinimumWidth = 60;
            this.Title.Name = "Title";
            this.Title.ReadOnly = true;
            this.Title.SortMode = System.Windows.Forms.DataGridViewColumnSortMode.NotSortable;
            this.Title.Width = 380;
            // 
            // SkipChapters
            // 
            this.SkipChapters.AutoSizeMode = System.Windows.Forms.DataGridViewAutoSizeColumnMode.None;
            dataGridViewCellStyle3.NullValue = null;
            this.SkipChapters.DefaultCellStyle = dataGridViewCellStyle3;
            this.SkipChapters.FillWeight = 1F;
            this.SkipChapters.HeaderText = "Skip Chapters";
            this.SkipChapters.MinimumWidth = 50;
            this.SkipChapters.Name = "SkipChapters";
            this.SkipChapters.SortMode = System.Windows.Forms.DataGridViewColumnSortMode.NotSortable;
            this.SkipChapters.Width = 110;
            // 
            // EndChapter
            // 
            this.EndChapter.AutoSizeMode = System.Windows.Forms.DataGridViewAutoSizeColumnMode.None;
            dataGridViewCellStyle4.NullValue = null;
            this.EndChapter.DefaultCellStyle = dataGridViewCellStyle4;
            this.EndChapter.FillWeight = 1F;
            this.EndChapter.HeaderText = "End Chapter";
            this.EndChapter.MaxInputLength = 2;
            this.EndChapter.MinimumWidth = 50;
            this.EndChapter.Name = "EndChapter";
            this.EndChapter.SortMode = System.Windows.Forms.DataGridViewColumnSortMode.NotSortable;
            // 
            // dgv_PlaylistContextMenu
            // 
            this.dgv_PlaylistContextMenu.ImageScalingSize = new System.Drawing.Size(20, 20);
            this.dgv_PlaylistContextMenu.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.playFileToolStripMenuItem,
            this.addFilesToolStripMenuItem,
            this.addFromClipboardToolStripMenuItem,
            this.removeFilesToolStripMenuItem,
            this.toolStripMenuItem4,
            this.addFolderToolStripMenuItem,
            this.openFolderToolStripMenuItem,
            this.toolStripMenuItem1,
            this.openPlaylistToolStripMenuItem,
            this.newPlaylistToolStripMenuItem,
            this.savePlaylistToolStripMenuItem,
            this.toolStripMenuItem2,
            this.nextItemToolStripMenuItem,
            this.previousItemToolStripMenuItem,
            this.toolStripMenuItem3,
            this.sortToolStripMenuItem,
            this.shuffleToolStripMenuItem});
            this.dgv_PlaylistContextMenu.Name = "contextMenuStrip1";
            this.dgv_PlaylistContextMenu.Size = new System.Drawing.Size(211, 314);
            // 
            // playFileToolStripMenuItem
            // 
            this.playFileToolStripMenuItem.Name = "playFileToolStripMenuItem";
            this.playFileToolStripMenuItem.Size = new System.Drawing.Size(210, 22);
            this.playFileToolStripMenuItem.Text = "Play file";
            this.playFileToolStripMenuItem.Click += new System.EventHandler(this.ButtonPlayClick);
            // 
            // addFilesToolStripMenuItem
            // 
            this.addFilesToolStripMenuItem.Name = "addFilesToolStripMenuItem";
            this.addFilesToolStripMenuItem.Size = new System.Drawing.Size(210, 22);
            this.addFilesToolStripMenuItem.Text = "Add file(s)";
            this.addFilesToolStripMenuItem.Click += new System.EventHandler(this.ButtonAddClick);
            // 
            // addFromClipboardToolStripMenuItem
            // 
            this.addFromClipboardToolStripMenuItem.Name = "addFromClipboardToolStripMenuItem";
            this.addFromClipboardToolStripMenuItem.Size = new System.Drawing.Size(210, 22);
            this.addFromClipboardToolStripMenuItem.Text = "Add file(s) from clipboard";
            this.addFromClipboardToolStripMenuItem.Click += new System.EventHandler(this.ButtonAddFromClipboardClick);
            // 
            // removeFilesToolStripMenuItem
            // 
            this.removeFilesToolStripMenuItem.Name = "removeFilesToolStripMenuItem";
            this.removeFilesToolStripMenuItem.Size = new System.Drawing.Size(210, 22);
            this.removeFilesToolStripMenuItem.Text = "Remove file(s)";
            this.removeFilesToolStripMenuItem.Click += new System.EventHandler(this.ButtonDelClick);
            // 
            // toolStripMenuItem4
            // 
            this.toolStripMenuItem4.Name = "toolStripMenuItem4";
            this.toolStripMenuItem4.Size = new System.Drawing.Size(207, 6);
            // 
            // addFolderToolStripMenuItem
            // 
            this.addFolderToolStripMenuItem.Name = "addFolderToolStripMenuItem";
            this.addFolderToolStripMenuItem.Size = new System.Drawing.Size(210, 22);
            this.addFolderToolStripMenuItem.Text = "Add folder";
            this.addFolderToolStripMenuItem.Click += new System.EventHandler(this.ButtonAddFolderClick);
            // 
            // openFolderToolStripMenuItem
            // 
            this.openFolderToolStripMenuItem.Name = "openFolderToolStripMenuItem";
            this.openFolderToolStripMenuItem.Size = new System.Drawing.Size(210, 22);
            this.openFolderToolStripMenuItem.Text = "Open folder";
            this.openFolderToolStripMenuItem.Click += new System.EventHandler(this.ButtonOpenFolderClick);
            // 
            // toolStripMenuItem1
            // 
            this.toolStripMenuItem1.Name = "toolStripMenuItem1";
            this.toolStripMenuItem1.Size = new System.Drawing.Size(207, 6);
            // 
            // openPlaylistToolStripMenuItem
            // 
            this.openPlaylistToolStripMenuItem.Name = "openPlaylistToolStripMenuItem";
            this.openPlaylistToolStripMenuItem.Size = new System.Drawing.Size(210, 22);
            this.openPlaylistToolStripMenuItem.Text = "Open playlist";
            this.openPlaylistToolStripMenuItem.Click += new System.EventHandler(this.ButtonOpenClick);
            // 
            // newPlaylistToolStripMenuItem
            // 
            this.newPlaylistToolStripMenuItem.Name = "newPlaylistToolStripMenuItem";
            this.newPlaylistToolStripMenuItem.Size = new System.Drawing.Size(210, 22);
            this.newPlaylistToolStripMenuItem.Text = "New playlist";
            this.newPlaylistToolStripMenuItem.Click += new System.EventHandler(this.ButtonNewClick);
            // 
            // savePlaylistToolStripMenuItem
            // 
            this.savePlaylistToolStripMenuItem.Name = "savePlaylistToolStripMenuItem";
            this.savePlaylistToolStripMenuItem.Size = new System.Drawing.Size(210, 22);
            this.savePlaylistToolStripMenuItem.Text = "Save playlist";
            this.savePlaylistToolStripMenuItem.Click += new System.EventHandler(this.ButtonSaveClick);
            // 
            // toolStripMenuItem2
            // 
            this.toolStripMenuItem2.Name = "toolStripMenuItem2";
            this.toolStripMenuItem2.Size = new System.Drawing.Size(207, 6);
            // 
            // nextItemToolStripMenuItem
            // 
            this.nextItemToolStripMenuItem.Name = "nextItemToolStripMenuItem";
            this.nextItemToolStripMenuItem.Size = new System.Drawing.Size(210, 22);
            this.nextItemToolStripMenuItem.Text = "Next";
            this.nextItemToolStripMenuItem.Click += new System.EventHandler(this.ButtonRightClick);
            // 
            // previousItemToolStripMenuItem
            // 
            this.previousItemToolStripMenuItem.Name = "previousItemToolStripMenuItem";
            this.previousItemToolStripMenuItem.Size = new System.Drawing.Size(210, 22);
            this.previousItemToolStripMenuItem.Text = "Previous";
            this.previousItemToolStripMenuItem.Click += new System.EventHandler(this.ButtonLeftClick);
            // 
            // toolStripMenuItem3
            // 
            this.toolStripMenuItem3.Name = "toolStripMenuItem3";
            this.toolStripMenuItem3.Size = new System.Drawing.Size(207, 6);
            // 
            // sortToolStripMenuItem
            // 
            this.sortToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.ascendingToolStripMenuItem,
            this.descendingToolStripMenuItem,
            this.restoreToolStripMenuItem});
            this.sortToolStripMenuItem.Name = "sortToolStripMenuItem";
            this.sortToolStripMenuItem.Size = new System.Drawing.Size(210, 22);
            this.sortToolStripMenuItem.Text = "Sort";
            // 
            // ascendingToolStripMenuItem
            // 
            this.ascendingToolStripMenuItem.Name = "ascendingToolStripMenuItem";
            this.ascendingToolStripMenuItem.Size = new System.Drawing.Size(136, 22);
            this.ascendingToolStripMenuItem.Text = "Ascending";
            this.ascendingToolStripMenuItem.Click += new System.EventHandler(this.ButtonSortAscendingClick);
            // 
            // descendingToolStripMenuItem
            // 
            this.descendingToolStripMenuItem.Name = "descendingToolStripMenuItem";
            this.descendingToolStripMenuItem.Size = new System.Drawing.Size(136, 22);
            this.descendingToolStripMenuItem.Text = "Descending";
            this.descendingToolStripMenuItem.Click += new System.EventHandler(this.ButtonSortDescendingClick);
            // 
            // restoreToolStripMenuItem
            // 
            this.restoreToolStripMenuItem.Name = "restoreToolStripMenuItem";
            this.restoreToolStripMenuItem.Size = new System.Drawing.Size(136, 22);
            this.restoreToolStripMenuItem.Text = "Restore";
            this.restoreToolStripMenuItem.Click += new System.EventHandler(this.ButtonRestoreClick);
            // 
            // shuffleToolStripMenuItem
            // 
            this.shuffleToolStripMenuItem.Name = "shuffleToolStripMenuItem";
            this.shuffleToolStripMenuItem.Size = new System.Drawing.Size(210, 22);
            this.shuffleToolStripMenuItem.Text = "Shuffle";
            this.shuffleToolStripMenuItem.Click += new System.EventHandler(this.ButtonShuffleClick);
            // 
            // dgv_PlaylistColumnContextMenu
            // 
            this.dgv_PlaylistColumnContextMenu.ImageScalingSize = new System.Drawing.Size(20, 20);
            this.dgv_PlaylistColumnContextMenu.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.numberToolStripMenuItem,
            this.fullPathToolStripMenuItem,
            this.directoryToolStripMenuItem,
            this.skipChaptersToolStripMenuItem,
            this.endChapterToolStripMenuItem});
            this.dgv_PlaylistColumnContextMenu.Name = "dgv_PlaylistColumnContextMenu";
            this.dgv_PlaylistColumnContextMenu.Size = new System.Drawing.Size(147, 114);
            // 
            // numberToolStripMenuItem
            // 
            this.numberToolStripMenuItem.Checked = true;
            this.numberToolStripMenuItem.CheckOnClick = true;
            this.numberToolStripMenuItem.CheckState = System.Windows.Forms.CheckState.Checked;
            this.numberToolStripMenuItem.Name = "numberToolStripMenuItem";
            this.numberToolStripMenuItem.Size = new System.Drawing.Size(146, 22);
            this.numberToolStripMenuItem.Text = "#";
            this.numberToolStripMenuItem.Click += new System.EventHandler(this.UpdateColumns);
            // 
            // fullPathToolStripMenuItem
            // 
            this.fullPathToolStripMenuItem.Checked = true;
            this.fullPathToolStripMenuItem.CheckOnClick = true;
            this.fullPathToolStripMenuItem.CheckState = System.Windows.Forms.CheckState.Checked;
            this.fullPathToolStripMenuItem.Name = "fullPathToolStripMenuItem";
            this.fullPathToolStripMenuItem.Size = new System.Drawing.Size(146, 22);
            this.fullPathToolStripMenuItem.Text = "Full Path";
            this.fullPathToolStripMenuItem.Click += new System.EventHandler(this.UpdateColumns);
            // 
            // directoryToolStripMenuItem
            // 
            this.directoryToolStripMenuItem.CheckOnClick = true;
            this.directoryToolStripMenuItem.Name = "directoryToolStripMenuItem";
            this.directoryToolStripMenuItem.Size = new System.Drawing.Size(146, 22);
            this.directoryToolStripMenuItem.Text = "Directory";
            this.directoryToolStripMenuItem.Click += new System.EventHandler(this.UpdateColumns);
            // 
            // skipChaptersToolStripMenuItem
            // 
            this.skipChaptersToolStripMenuItem.Checked = true;
            this.skipChaptersToolStripMenuItem.CheckOnClick = true;
            this.skipChaptersToolStripMenuItem.CheckState = System.Windows.Forms.CheckState.Checked;
            this.skipChaptersToolStripMenuItem.Name = "skipChaptersToolStripMenuItem";
            this.skipChaptersToolStripMenuItem.Size = new System.Drawing.Size(146, 22);
            this.skipChaptersToolStripMenuItem.Text = "Skip Chapters";
            this.skipChaptersToolStripMenuItem.Click += new System.EventHandler(this.UpdateColumns);
            // 
            // endChapterToolStripMenuItem
            // 
            this.endChapterToolStripMenuItem.Checked = true;
            this.endChapterToolStripMenuItem.CheckOnClick = true;
            this.endChapterToolStripMenuItem.CheckState = System.Windows.Forms.CheckState.Checked;
            this.endChapterToolStripMenuItem.Name = "endChapterToolStripMenuItem";
            this.endChapterToolStripMenuItem.Size = new System.Drawing.Size(146, 22);
            this.endChapterToolStripMenuItem.Text = "End Chapter";
            this.endChapterToolStripMenuItem.Click += new System.EventHandler(this.UpdateColumns);
            // 
            // PlaylistForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.AutoValidate = System.Windows.Forms.AutoValidate.EnablePreventFocusChange;
            this.ClientSize = new System.Drawing.Size(795, 233);
            this.Controls.Add(this.dgv_PlayList);
            this.Controls.Add(this.statusStrip1);
            this.KeyPreview = true;
            this.MinimumSize = new System.Drawing.Size(369, 258);
            this.Name = "PlaylistForm";
            this.StartPosition = System.Windows.Forms.FormStartPosition.Manual;
            this.Text = "Playlist";
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.PlaylistFormClosing);
            this.KeyDown += new System.Windows.Forms.KeyEventHandler(this.FormKeyDown);
            this.statusStrip1.ResumeLayout(false);
            this.statusStrip1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.dgv_PlayList)).EndInit();
            this.dgv_PlaylistContextMenu.ResumeLayout(false);
            this.dgv_PlaylistColumnContextMenu.ResumeLayout(false);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.StatusStrip statusStrip1;
        private ButtonStripItem buttonAdd;
        private ButtonStripItem buttonDel;
        private ButtonStripItem buttonNew;
        private ButtonStripItem buttonOpen;
        private ButtonStripItem buttonSave;
        private System.Windows.Forms.Timer timer;
        private ButtonStripItem buttonLeft;
        private ButtonStripItem buttonRight;
        private System.Windows.Forms.OpenFileDialog openPlaylistDialog;
        private System.Windows.Forms.SaveFileDialog savePlaylistDialog;
        private System.Windows.Forms.DataGridView dgv_PlayList;
        private ButtonStripItem buttonSortAscending;
        private ButtonStripItem buttonSortDescending;
        private ButtonStripItem buttonNewDisabled;
        private ContextMenuStrip dgv_PlaylistContextMenu;
        private ToolStripMenuItem addFilesToolStripMenuItem;
        private ToolStripMenuItem removeFilesToolStripMenuItem;
        private ToolStripMenuItem newPlaylistToolStripMenuItem;
        private ToolStripMenuItem openPlaylistToolStripMenuItem;
        private ToolStripMenuItem savePlaylistToolStripMenuItem;
        private ToolStripSeparator toolStripMenuItem1;
        private ToolStripSeparator toolStripMenuItem2;
        private ToolStripMenuItem nextItemToolStripMenuItem;
        private ToolStripMenuItem previousItemToolStripMenuItem;
        private ButtonStripItem buttonSettings;
        public OpenFileDialog openFileDialog;
        private ToolStripMenuItem playFileToolStripMenuItem;
        private ButtonStripItem PlayButton;
        private ButtonStripItem buttonSaveDisabled;
        private ButtonStripItem buttonNewEnabled;
        private ButtonStripItem buttonSaveEnabled;
        private ButtonStripItem buttonLeftDisabled;
        private ButtonStripItem buttonRightDisabled;
        private ButtonStripItem buttonLeftEnabled;
        private ButtonStripItem buttonRightEnabled;
        private ButtonStripItem buttonDelDisabled;
        private ButtonStripItem buttonDelEnabled;
        private ContextMenuStrip dgv_PlaylistColumnContextMenu;
        private ToolStripMenuItem numberToolStripMenuItem;
        private ToolStripMenuItem skipChaptersToolStripMenuItem;
        private ToolStripMenuItem endChapterToolStripMenuItem;
        private ToolStripMenuItem fullPathToolStripMenuItem;
        private ToolStripMenuItem directoryToolStripMenuItem;
        private ButtonStripItem buttonShuffle;
        private ButtonStripItem buttonShuffleDisabled;
        private ButtonStripItem buttonSortDescendingDisabled;
        private ToolStripSeparator toolStripMenuItem3;
        private ToolStripMenuItem sortToolStripMenuItem;
        private ToolStripMenuItem ascendingToolStripMenuItem;
        private ToolStripMenuItem descendingToolStripMenuItem;
        private ToolStripMenuItem shuffleToolStripMenuItem;
        private ButtonStripItem buttonSortDescendingEnabled;
        private ButtonStripItem buttonSortAscendingEnabled;
        private ButtonStripItem buttonSortAscendingDisabled;
        private ButtonStripItem buttonShuffleEnabled;
        private ToolStripStatusLabel toolStripStatusLabel2;
        private ToolStripStatusLabel toolStripStatusLabel1;
        private ButtonStripItem buttonAddFolder;
        private ToolStripMenuItem addFolderToolStripMenuItem;
        private ToolStripMenuItem restoreToolStripMenuItem;
        private ToolStripMenuItem openFolderToolStripMenuItem;
        private ToolStripSeparator toolStripMenuItem4;
        private ButtonStripItem buttonRestoreEnabled;
        private ButtonStripItem buttonRestoreDisabled;
        private ButtonStripItem buttonRestore;
        private DataGridViewImageColumn Playing;
        private DataGridViewTextBoxColumn Number;
        private DataGridViewTextBoxColumn FullPath;
        private DataGridViewTextBoxColumn CurrentDirectory;
        private DataGridViewTextBoxColumn Title;
        private DataGridViewTextBoxColumn SkipChapters;
        private DataGridViewTextBoxColumn EndChapter;
        private ButtonStripItem PauseButton;
        private ButtonStripItem StopButton;
        private ToolStripMenuItem addFromClipboardToolStripMenuItem;
    }
}
