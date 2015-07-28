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
            this.statusStrip1 = new Mpdn.Extensions.PlayerExtensions.Playlist.CustomStatusStrip();
            this.buttonAdd = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonDel = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonAddFolder = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.toolStripStatusLabel2 = new System.Windows.Forms.ToolStripStatusLabel();
            this.buttonLeft = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonRight = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonSortAscending = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonSortDescending = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonShuffle = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonRepeatPlaylist = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonRestore = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.PlayButton = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.PauseButton = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.StopButton = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.toolStripStatusLabel1 = new System.Windows.Forms.ToolStripStatusLabel();
            this.buttonNewPlaylist = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonOpenPlaylist = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonSavePlaylist = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.buttonSettings = new Mpdn.Extensions.PlayerExtensions.Playlist.ButtonStripItem();
            this.timer = new System.Windows.Forms.Timer(this.components);
            this.openFileDialog = new System.Windows.Forms.OpenFileDialog();
            this.openPlaylistDialog = new System.Windows.Forms.OpenFileDialog();
            this.savePlaylistDialog = new System.Windows.Forms.SaveFileDialog();
            this.dgv_PlayList = new Mpdn.Extensions.PlayerExtensions.Playlist.PlaylistDataGrid();
            this.Playing = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.Number = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.FullPath = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.CurrentDirectory = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.Title = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.SkipChapters = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.EndChapter = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.Duration = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.dgv_PlaylistContextMenu = new System.Windows.Forms.ContextMenuStrip(this.components);
            this.playToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.addToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.openFilesToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.addFolderToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.addClipboardToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.addPlaylistToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.openToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.openFileToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.openFolderToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.openClipboardToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.openPlaylistToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.removeToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.removeSelectedItemsToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.removeUnselectedItemsToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.removeNonExistentItemsToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.saveToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.newPlaylistToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.savePlaylistToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.savePlaylistAsToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.viewToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.viewFileLocationToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.viewMediaInfoToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.sortToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.ascendingToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.descendingToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.shuffleToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.restoreToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.dgv_PlaylistColumnContextMenu = new System.Windows.Forms.ContextMenuStrip(this.components);
            this.numberToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.fullPathToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.directoryToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.skipChaptersToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.endChapterToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.durationToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.statusStrip1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.dgv_PlayList)).BeginInit();
            this.dgv_PlaylistContextMenu.SuspendLayout();
            this.dgv_PlaylistColumnContextMenu.SuspendLayout();
            this.SuspendLayout();
            // 
            // statusStrip1
            // 
            this.statusStrip1.BackColor = System.Drawing.Color.Transparent;
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
            this.buttonRepeatPlaylist,
            this.buttonRestore,
            this.PlayButton,
            this.PauseButton,
            this.StopButton,
            this.toolStripStatusLabel1,
            this.buttonNewPlaylist,
            this.buttonOpenPlaylist,
            this.buttonSavePlaylist,
            this.buttonSettings});
            this.statusStrip1.Location = new System.Drawing.Point(0, 206);
            this.statusStrip1.Name = "statusStrip1";
            this.statusStrip1.Padding = new System.Windows.Forms.Padding(1, 0, 15, 0);
            this.statusStrip1.RenderMode = System.Windows.Forms.ToolStripRenderMode.ManagerRenderMode;
            this.statusStrip1.ShowItemToolTips = true;
            this.statusStrip1.Size = new System.Drawing.Size(684, 27);
            this.statusStrip1.TabIndex = 0;
            this.statusStrip1.TabStop = true;
            this.statusStrip1.Text = "statusStrip1";
            this.statusStrip1.Visible = false;
            // 
            // buttonAdd
            // 
            this.buttonAdd.AutoSize = false;
            this.buttonAdd.BackColor = System.Drawing.Color.Transparent;
            this.buttonAdd.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonAdd.Name = "buttonAdd";
            this.buttonAdd.Size = new System.Drawing.Size(24, 25);
            this.buttonAdd.ToolTipText = "Add file(s)";
            this.buttonAdd.Visible = false;
            this.buttonAdd.Click += new System.EventHandler(this.ButtonAddFilesClick);
            // 
            // buttonDel
            // 
            this.buttonDel.AutoSize = false;
            this.buttonDel.BackColor = System.Drawing.Color.Transparent;
            this.buttonDel.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonDel.Name = "buttonDel";
            this.buttonDel.Size = new System.Drawing.Size(24, 25);
            this.buttonDel.ToolTipText = "Remove file(s)";
            this.buttonDel.Visible = false;
            this.buttonDel.Click += new System.EventHandler(this.ButtonRemoveSelectedItemsClick);
            // 
            // buttonAddFolder
            // 
            this.buttonAddFolder.AutoSize = false;
            this.buttonAddFolder.BackColor = System.Drawing.Color.Transparent;
            this.buttonAddFolder.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonAddFolder.Name = "buttonAddFolder";
            this.buttonAddFolder.Size = new System.Drawing.Size(24, 25);
            this.buttonAddFolder.ToolTipText = "Add folder";
            this.buttonAddFolder.Visible = false;
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
            this.buttonLeft.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonLeft.Name = "buttonLeft";
            this.buttonLeft.Size = new System.Drawing.Size(24, 25);
            this.buttonLeft.ToolTipText = "Previous";
            this.buttonLeft.Visible = false;
            this.buttonLeft.Click += new System.EventHandler(this.ButtonLeftClick);
            // 
            // buttonRight
            // 
            this.buttonRight.AutoSize = false;
            this.buttonRight.BackColor = System.Drawing.Color.Transparent;
            this.buttonRight.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonRight.Name = "buttonRight";
            this.buttonRight.Size = new System.Drawing.Size(24, 25);
            this.buttonRight.ToolTipText = "Next";
            this.buttonRight.Visible = false;
            this.buttonRight.Click += new System.EventHandler(this.ButtonRightClick);
            // 
            // buttonSortAscending
            // 
            this.buttonSortAscending.AutoSize = false;
            this.buttonSortAscending.BackColor = System.Drawing.Color.Transparent;
            this.buttonSortAscending.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonSortAscending.Margin = new System.Windows.Forms.Padding(10, 2, 0, 0);
            this.buttonSortAscending.Name = "buttonSortAscending";
            this.buttonSortAscending.Size = new System.Drawing.Size(24, 25);
            this.buttonSortAscending.ToolTipText = "Sort playlist (ascending)";
            this.buttonSortAscending.Visible = false;
            this.buttonSortAscending.Click += new System.EventHandler(this.ButtonSortAscendingClick);
            // 
            // buttonSortDescending
            // 
            this.buttonSortDescending.AutoSize = false;
            this.buttonSortDescending.BackColor = System.Drawing.Color.Transparent;
            this.buttonSortDescending.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonSortDescending.Name = "buttonSortDescending";
            this.buttonSortDescending.Size = new System.Drawing.Size(24, 25);
            this.buttonSortDescending.ToolTipText = "Sort playlist (descending)";
            this.buttonSortDescending.Visible = false;
            this.buttonSortDescending.Click += new System.EventHandler(this.ButtonSortDescendingClick);
            // 
            // buttonShuffle
            // 
            this.buttonShuffle.AutoSize = false;
            this.buttonShuffle.BackColor = System.Drawing.Color.Transparent;
            this.buttonShuffle.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonShuffle.Name = "buttonShuffle";
            this.buttonShuffle.Size = new System.Drawing.Size(24, 25);
            this.buttonShuffle.ToolTipText = "Shuffle playlist";
            this.buttonShuffle.Visible = false;
            this.buttonShuffle.Click += new System.EventHandler(this.ButtonShuffleClick);
            // 
            // buttonRepeatPlaylist
            // 
            this.buttonRepeatPlaylist.AutoSize = false;
            this.buttonRepeatPlaylist.BackColor = System.Drawing.Color.Transparent;
            this.buttonRepeatPlaylist.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonRepeatPlaylist.Name = "buttonRepeatPlaylist";
            this.buttonRepeatPlaylist.Size = new System.Drawing.Size(24, 25);
            this.buttonRepeatPlaylist.ToolTipText = "Repeat playlist";
            this.buttonRepeatPlaylist.Visible = false;
            this.buttonRepeatPlaylist.Click += new System.EventHandler(this.ButtonRepeatPlaylistClick);
            // 
            // buttonRestore
            // 
            this.buttonRestore.AutoSize = false;
            this.buttonRestore.BackColor = System.Drawing.Color.Transparent;
            this.buttonRestore.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonRestore.Name = "buttonRestore";
            this.buttonRestore.Size = new System.Drawing.Size(24, 25);
            this.buttonRestore.ToolTipText = "Restore playlist";
            this.buttonRestore.Visible = false;
            this.buttonRestore.Click += new System.EventHandler(this.ButtonRestoreClick);
            // 
            // PlayButton
            // 
            this.PlayButton.AutoSize = false;
            this.PlayButton.BackColor = System.Drawing.Color.Transparent;
            this.PlayButton.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("PlayButton.BackgroundImage")));
            this.PlayButton.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.PlayButton.Name = "PlayButton";
            this.PlayButton.Size = new System.Drawing.Size(24, 25);
            this.PlayButton.Visible = false;
            // 
            // PauseButton
            // 
            this.PauseButton.AutoSize = false;
            this.PauseButton.BackColor = System.Drawing.Color.Transparent;
            this.PauseButton.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("PauseButton.BackgroundImage")));
            this.PauseButton.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.PauseButton.Name = "PauseButton";
            this.PauseButton.Size = new System.Drawing.Size(24, 25);
            this.PauseButton.Visible = false;
            // 
            // StopButton
            // 
            this.StopButton.AutoSize = false;
            this.StopButton.BackColor = System.Drawing.Color.Transparent;
            this.StopButton.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("StopButton.BackgroundImage")));
            this.StopButton.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.StopButton.Name = "StopButton";
            this.StopButton.Size = new System.Drawing.Size(24, 25);
            this.StopButton.Visible = false;
            // 
            // toolStripStatusLabel1
            // 
            this.toolStripStatusLabel1.Name = "toolStripStatusLabel1";
            this.toolStripStatusLabel1.Size = new System.Drawing.Size(658, 22);
            this.toolStripStatusLabel1.Spring = true;
            // 
            // buttonNewPlaylist
            // 
            this.buttonNewPlaylist.AutoSize = false;
            this.buttonNewPlaylist.BackColor = System.Drawing.Color.Transparent;
            this.buttonNewPlaylist.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonNewPlaylist.Name = "buttonNewPlaylist";
            this.buttonNewPlaylist.Size = new System.Drawing.Size(24, 25);
            this.buttonNewPlaylist.ToolTipText = "New playlist";
            this.buttonNewPlaylist.Visible = false;
            this.buttonNewPlaylist.Click += new System.EventHandler(this.ButtonNewPlaylistClick);
            // 
            // buttonOpenPlaylist
            // 
            this.buttonOpenPlaylist.AutoSize = false;
            this.buttonOpenPlaylist.BackColor = System.Drawing.Color.Transparent;
            this.buttonOpenPlaylist.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonOpenPlaylist.Name = "buttonOpenPlaylist";
            this.buttonOpenPlaylist.Size = new System.Drawing.Size(24, 25);
            this.buttonOpenPlaylist.ToolTipText = "Open playlist";
            this.buttonOpenPlaylist.Visible = false;
            this.buttonOpenPlaylist.Click += new System.EventHandler(this.ButtonOpenPlaylistClick);
            // 
            // buttonSavePlaylist
            // 
            this.buttonSavePlaylist.AutoSize = false;
            this.buttonSavePlaylist.BackColor = System.Drawing.Color.Transparent;
            this.buttonSavePlaylist.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonSavePlaylist.Name = "buttonSavePlaylist";
            this.buttonSavePlaylist.Size = new System.Drawing.Size(24, 25);
            this.buttonSavePlaylist.ToolTipText = "Save playlist";
            this.buttonSavePlaylist.Visible = false;
            this.buttonSavePlaylist.Click += new System.EventHandler(this.ButtonSavePlaylistAsClick);
            // 
            // buttonSettings
            // 
            this.buttonSettings.BackColor = System.Drawing.Color.Transparent;
            this.buttonSettings.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.buttonSettings.Name = "buttonSettings";
            this.buttonSettings.Size = new System.Drawing.Size(24, 25);
            this.buttonSettings.ToolTipText = "Configure playlist";
            this.buttonSettings.Visible = false;
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
            this.dgv_PlayList.BackgroundColor = System.Drawing.SystemColors.Window;
            this.dgv_PlayList.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.dgv_PlayList.CellBorderStyle = System.Windows.Forms.DataGridViewCellBorderStyle.None;
            this.dgv_PlayList.ColumnHeadersHeight = 21;
            this.dgv_PlayList.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.DisableResizing;
            this.dgv_PlayList.Columns.AddRange(new System.Windows.Forms.DataGridViewColumn[] {
            this.Playing,
            this.Number,
            this.FullPath,
            this.CurrentDirectory,
            this.Title,
            this.SkipChapters,
            this.EndChapter,
            this.Duration});
            dataGridViewCellStyle5.Alignment = System.Windows.Forms.DataGridViewContentAlignment.MiddleLeft;
            dataGridViewCellStyle5.BackColor = System.Drawing.Color.White;
            dataGridViewCellStyle5.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            dataGridViewCellStyle5.ForeColor = System.Drawing.Color.Black;
            dataGridViewCellStyle5.SelectionBackColor = System.Drawing.Color.SkyBlue;
            dataGridViewCellStyle5.SelectionForeColor = System.Drawing.Color.White;
            dataGridViewCellStyle5.WrapMode = System.Windows.Forms.DataGridViewTriState.False;
            this.dgv_PlayList.DefaultCellStyle = dataGridViewCellStyle5;
            this.dgv_PlayList.Dock = System.Windows.Forms.DockStyle.Fill;
            this.dgv_PlayList.EditMode = System.Windows.Forms.DataGridViewEditMode.EditOnEnter;
            this.dgv_PlayList.GridColor = System.Drawing.SystemColors.Window;
            this.dgv_PlayList.Location = new System.Drawing.Point(0, 0);
            this.dgv_PlayList.Name = "dgv_PlayList";
            this.dgv_PlayList.RowHeadersVisible = false;
            this.dgv_PlayList.RowHeadersWidth = 4;
            this.dgv_PlayList.RowTemplate.Height = 29;
            this.dgv_PlayList.RowTemplate.Resizable = System.Windows.Forms.DataGridViewTriState.False;
            this.dgv_PlayList.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
            this.dgv_PlayList.SelectionMode = System.Windows.Forms.DataGridViewSelectionMode.FullRowSelect;
            this.dgv_PlayList.ShowCellErrors = false;
            this.dgv_PlayList.ShowCellToolTips = false;
            this.dgv_PlayList.ShowEditingIcon = false;
            this.dgv_PlayList.ShowRowErrors = false;
            this.dgv_PlayList.Size = new System.Drawing.Size(684, 233);
            this.dgv_PlayList.TabIndex = 1;
            this.dgv_PlayList.TabStop = false;
            // 
            // Playing
            // 
            this.Playing.AutoSizeMode = System.Windows.Forms.DataGridViewAutoSizeColumnMode.None;
            dataGridViewCellStyle2.NullValue = "null";
            this.Playing.DefaultCellStyle = dataGridViewCellStyle2;
            this.Playing.FillWeight = 1F;
            this.Playing.HeaderText = "";
            this.Playing.MinimumWidth = 24;
            this.Playing.Name = "Playing";
            this.Playing.ReadOnly = true;
            this.Playing.Resizable = System.Windows.Forms.DataGridViewTriState.False;
            this.Playing.SortMode = System.Windows.Forms.DataGridViewColumnSortMode.NotSortable;
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
            this.FullPath.FillWeight = 1F;
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
            this.CurrentDirectory.FillWeight = 1F;
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
            this.Title.FillWeight = 1F;
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
            this.SkipChapters.Visible = false;
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
            this.EndChapter.Visible = false;
            // 
            // Duration
            // 
            this.Duration.AutoSizeMode = System.Windows.Forms.DataGridViewAutoSizeColumnMode.None;
            this.Duration.FillWeight = 1F;
            this.Duration.HeaderText = "Duration";
            this.Duration.MinimumWidth = 40;
            this.Duration.Name = "Duration";
            this.Duration.ReadOnly = true;
            this.Duration.SortMode = System.Windows.Forms.DataGridViewColumnSortMode.NotSortable;
            // 
            // dgv_PlaylistContextMenu
            // 
            this.dgv_PlaylistContextMenu.ImageScalingSize = new System.Drawing.Size(20, 20);
            this.dgv_PlaylistContextMenu.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.playToolStripMenuItem,
            this.addToolStripMenuItem,
            this.openToolStripMenuItem,
            this.removeToolStripMenuItem,
            this.saveToolStripMenuItem,
            this.viewToolStripMenuItem,
            this.sortToolStripMenuItem});
            this.dgv_PlaylistContextMenu.Name = "contextMenuStrip1";
            this.dgv_PlaylistContextMenu.Size = new System.Drawing.Size(118, 158);
            // 
            // playToolStripMenuItem
            // 
            this.playToolStripMenuItem.Name = "playToolStripMenuItem";
            this.playToolStripMenuItem.Size = new System.Drawing.Size(117, 22);
            this.playToolStripMenuItem.Text = "Play";
            this.playToolStripMenuItem.Click += new System.EventHandler(this.ButtonPlayClick);
            // 
            // addToolStripMenuItem
            // 
            this.addToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.openFilesToolStripMenuItem,
            this.addFolderToolStripMenuItem,
            this.addClipboardToolStripMenuItem,
            this.addPlaylistToolStripMenuItem});
            this.addToolStripMenuItem.Name = "addToolStripMenuItem";
            this.addToolStripMenuItem.Size = new System.Drawing.Size(117, 22);
            this.addToolStripMenuItem.Text = "Add";
            // 
            // openFilesToolStripMenuItem
            // 
            this.openFilesToolStripMenuItem.Name = "openFilesToolStripMenuItem";
            this.openFilesToolStripMenuItem.Size = new System.Drawing.Size(155, 22);
            this.openFilesToolStripMenuItem.Text = "File(s)";
            this.openFilesToolStripMenuItem.Click += new System.EventHandler(this.ButtonAddFilesClick);
            // 
            // addFolderToolStripMenuItem
            // 
            this.addFolderToolStripMenuItem.Name = "addFolderToolStripMenuItem";
            this.addFolderToolStripMenuItem.Size = new System.Drawing.Size(155, 22);
            this.addFolderToolStripMenuItem.Text = "Folder";
            this.addFolderToolStripMenuItem.Click += new System.EventHandler(this.ButtonAddFolderClick);
            // 
            // addClipboardToolStripMenuItem
            // 
            this.addClipboardToolStripMenuItem.Name = "addClipboardToolStripMenuItem";
            this.addClipboardToolStripMenuItem.Size = new System.Drawing.Size(155, 22);
            this.addClipboardToolStripMenuItem.Text = "From clipboard";
            this.addClipboardToolStripMenuItem.Click += new System.EventHandler(this.ButtonAddFromClipboardClick);
            // 
            // addPlaylistToolStripMenuItem
            // 
            this.addPlaylistToolStripMenuItem.Name = "addPlaylistToolStripMenuItem";
            this.addPlaylistToolStripMenuItem.Size = new System.Drawing.Size(155, 22);
            this.addPlaylistToolStripMenuItem.Text = "Playlist";
            this.addPlaylistToolStripMenuItem.Click += new System.EventHandler(this.ButtonAddPlaylistClick);
            // 
            // openToolStripMenuItem
            // 
            this.openToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.openFileToolStripMenuItem,
            this.openFolderToolStripMenuItem,
            this.openClipboardToolStripMenuItem,
            this.openPlaylistToolStripMenuItem});
            this.openToolStripMenuItem.Name = "openToolStripMenuItem";
            this.openToolStripMenuItem.Size = new System.Drawing.Size(117, 22);
            this.openToolStripMenuItem.Text = "Open";
            // 
            // openFileToolStripMenuItem
            // 
            this.openFileToolStripMenuItem.Name = "openFileToolStripMenuItem";
            this.openFileToolStripMenuItem.Size = new System.Drawing.Size(155, 22);
            this.openFileToolStripMenuItem.Text = "File(s)";
            this.openFileToolStripMenuItem.Click += new System.EventHandler(this.ButtonOpenFilesClick);
            // 
            // openFolderToolStripMenuItem
            // 
            this.openFolderToolStripMenuItem.Name = "openFolderToolStripMenuItem";
            this.openFolderToolStripMenuItem.Size = new System.Drawing.Size(155, 22);
            this.openFolderToolStripMenuItem.Text = "Folder";
            this.openFolderToolStripMenuItem.Click += new System.EventHandler(this.ButtonOpenFolderClick);
            // 
            // openClipboardToolStripMenuItem
            // 
            this.openClipboardToolStripMenuItem.Name = "openClipboardToolStripMenuItem";
            this.openClipboardToolStripMenuItem.Size = new System.Drawing.Size(155, 22);
            this.openClipboardToolStripMenuItem.Text = "From clipboard";
            this.openClipboardToolStripMenuItem.Click += new System.EventHandler(this.ButtonOpenFromClipboardClick);
            // 
            // openPlaylistToolStripMenuItem
            // 
            this.openPlaylistToolStripMenuItem.Name = "openPlaylistToolStripMenuItem";
            this.openPlaylistToolStripMenuItem.Size = new System.Drawing.Size(155, 22);
            this.openPlaylistToolStripMenuItem.Text = "Playlist";
            this.openPlaylistToolStripMenuItem.Click += new System.EventHandler(this.ButtonOpenPlaylistClick);
            // 
            // removeToolStripMenuItem
            // 
            this.removeToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.removeSelectedItemsToolStripMenuItem,
            this.removeUnselectedItemsToolStripMenuItem,
            this.removeNonExistentItemsToolStripMenuItem});
            this.removeToolStripMenuItem.Name = "removeToolStripMenuItem";
            this.removeToolStripMenuItem.Size = new System.Drawing.Size(117, 22);
            this.removeToolStripMenuItem.Text = "Remove";
            // 
            // removeSelectedItemsToolStripMenuItem
            // 
            this.removeSelectedItemsToolStripMenuItem.Name = "removeSelectedItemsToolStripMenuItem";
            this.removeSelectedItemsToolStripMenuItem.Size = new System.Drawing.Size(169, 22);
            this.removeSelectedItemsToolStripMenuItem.Text = "Selected items";
            this.removeSelectedItemsToolStripMenuItem.Click += new System.EventHandler(this.ButtonRemoveSelectedItemsClick);
            // 
            // removeUnselectedItemsToolStripMenuItem
            // 
            this.removeUnselectedItemsToolStripMenuItem.Name = "removeUnselectedItemsToolStripMenuItem";
            this.removeUnselectedItemsToolStripMenuItem.Size = new System.Drawing.Size(169, 22);
            this.removeUnselectedItemsToolStripMenuItem.Text = "Unselected items";
            this.removeUnselectedItemsToolStripMenuItem.Click += new System.EventHandler(this.ButtonRemoveUnselectedItemsClick);
            // 
            // removeNonExistentItemsToolStripMenuItem
            // 
            this.removeNonExistentItemsToolStripMenuItem.Name = "removeNonExistentItemsToolStripMenuItem";
            this.removeNonExistentItemsToolStripMenuItem.Size = new System.Drawing.Size(169, 22);
            this.removeNonExistentItemsToolStripMenuItem.Text = "Nonexistent items";
            this.removeNonExistentItemsToolStripMenuItem.Click += new System.EventHandler(this.ButtonRemoveNonExistentItemsClick);
            // 
            // saveToolStripMenuItem
            // 
            this.saveToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.newPlaylistToolStripMenuItem,
            this.savePlaylistToolStripMenuItem,
            this.savePlaylistAsToolStripMenuItem});
            this.saveToolStripMenuItem.Name = "saveToolStripMenuItem";
            this.saveToolStripMenuItem.Size = new System.Drawing.Size(117, 22);
            this.saveToolStripMenuItem.Text = "Playlist";
            // 
            // newPlaylistToolStripMenuItem
            // 
            this.newPlaylistToolStripMenuItem.Name = "newPlaylistToolStripMenuItem";
            this.newPlaylistToolStripMenuItem.Size = new System.Drawing.Size(152, 22);
            this.newPlaylistToolStripMenuItem.Text = "New playlist";
            this.newPlaylistToolStripMenuItem.Click += new System.EventHandler(this.ButtonNewPlaylistClick);
            // 
            // savePlaylistToolStripMenuItem
            // 
            this.savePlaylistToolStripMenuItem.Name = "savePlaylistToolStripMenuItem";
            this.savePlaylistToolStripMenuItem.Size = new System.Drawing.Size(152, 22);
            this.savePlaylistToolStripMenuItem.Text = "Save playlist";
            this.savePlaylistToolStripMenuItem.Click += new System.EventHandler(this.ButtonSavePlaylistClick);
            // 
            // savePlaylistAsToolStripMenuItem
            // 
            this.savePlaylistAsToolStripMenuItem.Name = "savePlaylistAsToolStripMenuItem";
            this.savePlaylistAsToolStripMenuItem.Size = new System.Drawing.Size(152, 22);
            this.savePlaylistAsToolStripMenuItem.Text = "Save playlist as";
            this.savePlaylistAsToolStripMenuItem.Click += new System.EventHandler(this.ButtonSavePlaylistAsClick);
            // 
            // viewToolStripMenuItem
            // 
            this.viewToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.viewFileLocationToolStripMenuItem,
            this.viewMediaInfoToolStripMenuItem});
            this.viewToolStripMenuItem.Name = "viewToolStripMenuItem";
            this.viewToolStripMenuItem.Size = new System.Drawing.Size(117, 22);
            this.viewToolStripMenuItem.Text = "View";
            // 
            // viewFileLocationToolStripMenuItem
            // 
            this.viewFileLocationToolStripMenuItem.Name = "viewFileLocationToolStripMenuItem";
            this.viewFileLocationToolStripMenuItem.Size = new System.Drawing.Size(138, 22);
            this.viewFileLocationToolStripMenuItem.Text = "File location";
            this.viewFileLocationToolStripMenuItem.Click += new System.EventHandler(this.ButtonViewFileLocation);
            // 
            // viewMediaInfoToolStripMenuItem
            // 
            this.viewMediaInfoToolStripMenuItem.Name = "viewMediaInfoToolStripMenuItem";
            this.viewMediaInfoToolStripMenuItem.Size = new System.Drawing.Size(138, 22);
            this.viewMediaInfoToolStripMenuItem.Text = "Media info";
            this.viewMediaInfoToolStripMenuItem.Click += new System.EventHandler(this.ButtonViewMediaInfo);
            // 
            // sortToolStripMenuItem
            // 
            this.sortToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.ascendingToolStripMenuItem,
            this.descendingToolStripMenuItem,
            this.shuffleToolStripMenuItem,
            this.restoreToolStripMenuItem});
            this.sortToolStripMenuItem.Name = "sortToolStripMenuItem";
            this.sortToolStripMenuItem.Size = new System.Drawing.Size(117, 22);
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
            // shuffleToolStripMenuItem
            // 
            this.shuffleToolStripMenuItem.Name = "shuffleToolStripMenuItem";
            this.shuffleToolStripMenuItem.Size = new System.Drawing.Size(136, 22);
            this.shuffleToolStripMenuItem.Text = "Shuffle";
            this.shuffleToolStripMenuItem.Click += new System.EventHandler(this.ButtonShuffleClick);
            // 
            // restoreToolStripMenuItem
            // 
            this.restoreToolStripMenuItem.Name = "restoreToolStripMenuItem";
            this.restoreToolStripMenuItem.Size = new System.Drawing.Size(136, 22);
            this.restoreToolStripMenuItem.Text = "Restore";
            this.restoreToolStripMenuItem.Click += new System.EventHandler(this.ButtonRestoreClick);
            // 
            // dgv_PlaylistColumnContextMenu
            // 
            this.dgv_PlaylistColumnContextMenu.ImageScalingSize = new System.Drawing.Size(20, 20);
            this.dgv_PlaylistColumnContextMenu.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.numberToolStripMenuItem,
            this.fullPathToolStripMenuItem,
            this.directoryToolStripMenuItem,
            this.skipChaptersToolStripMenuItem,
            this.endChapterToolStripMenuItem,
            this.durationToolStripMenuItem});
            this.dgv_PlaylistColumnContextMenu.Name = "dgv_PlaylistColumnContextMenu";
            this.dgv_PlaylistColumnContextMenu.Size = new System.Drawing.Size(147, 136);
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
            // durationToolStripMenuItem
            // 
            this.durationToolStripMenuItem.CheckOnClick = true;
            this.durationToolStripMenuItem.Name = "durationToolStripMenuItem";
            this.durationToolStripMenuItem.Size = new System.Drawing.Size(146, 22);
            this.durationToolStripMenuItem.Text = "Duration";
            this.durationToolStripMenuItem.Click += new System.EventHandler(this.UpdateColumns);
            // 
            // PlaylistForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(96F, 96F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Dpi;
            this.AutoValidate = System.Windows.Forms.AutoValidate.EnablePreventFocusChange;
            this.ClientSize = new System.Drawing.Size(684, 233);
            this.Controls.Add(this.dgv_PlayList);
            this.Controls.Add(this.statusStrip1);
            this.KeyPreview = true;
            this.MinimumSize = new System.Drawing.Size(395, 115);
            this.Name = "PlaylistForm";
            this.StartPosition = System.Windows.Forms.FormStartPosition.Manual;
            this.Text = "Playlist";
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.PlaylistFormClosing);
            this.KeyDown += new System.Windows.Forms.KeyEventHandler(this.PlaylistForm_KeyDown);
            this.statusStrip1.ResumeLayout(false);
            this.statusStrip1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.dgv_PlayList)).EndInit();
            this.dgv_PlaylistContextMenu.ResumeLayout(false);
            this.dgv_PlaylistColumnContextMenu.ResumeLayout(false);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private CustomStatusStrip statusStrip1;
        private ButtonStripItem buttonAdd;
        private ButtonStripItem buttonDel;
        private ButtonStripItem buttonNewPlaylist;
        private ButtonStripItem buttonOpenPlaylist;
        private ButtonStripItem buttonSavePlaylist;
        private System.Windows.Forms.Timer timer;
        private ButtonStripItem buttonLeft;
        private ButtonStripItem buttonRight;
        private System.Windows.Forms.OpenFileDialog openPlaylistDialog;
        private System.Windows.Forms.SaveFileDialog savePlaylistDialog;
        private PlaylistDataGrid dgv_PlayList;
        private ButtonStripItem buttonSortAscending;
        private ButtonStripItem buttonSortDescending;
        private ContextMenuStrip dgv_PlaylistContextMenu;
        private ButtonStripItem buttonSettings;
        public OpenFileDialog openFileDialog;
        private ToolStripMenuItem playToolStripMenuItem;
        private ButtonStripItem PlayButton;
        private ContextMenuStrip dgv_PlaylistColumnContextMenu;
        private ToolStripMenuItem numberToolStripMenuItem;
        private ToolStripMenuItem skipChaptersToolStripMenuItem;
        private ToolStripMenuItem endChapterToolStripMenuItem;
        private ToolStripMenuItem fullPathToolStripMenuItem;
        private ToolStripMenuItem directoryToolStripMenuItem;
        private ButtonStripItem buttonShuffle;
        private ToolStripMenuItem sortToolStripMenuItem;
        private ToolStripMenuItem ascendingToolStripMenuItem;
        private ToolStripMenuItem descendingToolStripMenuItem;
        private ToolStripStatusLabel toolStripStatusLabel2;
        private ToolStripStatusLabel toolStripStatusLabel1;
        private ButtonStripItem buttonAddFolder;
        private ToolStripMenuItem restoreToolStripMenuItem;
        private ButtonStripItem buttonRestore;
        private ButtonStripItem PauseButton;
        private ButtonStripItem StopButton;
        private ToolStripMenuItem addToolStripMenuItem;
        private ToolStripMenuItem openFilesToolStripMenuItem;
        private ToolStripMenuItem addFolderToolStripMenuItem;
        private ToolStripMenuItem addClipboardToolStripMenuItem;
        private ToolStripMenuItem openToolStripMenuItem;
        private ToolStripMenuItem saveToolStripMenuItem;
        private ToolStripMenuItem savePlaylistToolStripMenuItem;
        private ToolStripMenuItem removeToolStripMenuItem;
        private ToolStripMenuItem removeSelectedItemsToolStripMenuItem;
        private ToolStripMenuItem shuffleToolStripMenuItem;
        private ToolStripMenuItem openFileToolStripMenuItem;
        private ToolStripMenuItem openFolderToolStripMenuItem;
        private ToolStripMenuItem openClipboardToolStripMenuItem;
        private ToolStripMenuItem openPlaylistToolStripMenuItem;
        private ToolStripMenuItem viewToolStripMenuItem;
        private ToolStripMenuItem viewFileLocationToolStripMenuItem;
        private ToolStripMenuItem viewMediaInfoToolStripMenuItem;
        private ToolStripMenuItem newPlaylistToolStripMenuItem;
        private ToolStripMenuItem savePlaylistAsToolStripMenuItem;
        private ToolStripMenuItem removeUnselectedItemsToolStripMenuItem;
        private ToolStripMenuItem removeNonExistentItemsToolStripMenuItem;
        private ToolStripMenuItem durationToolStripMenuItem;
        private ToolStripMenuItem addPlaylistToolStripMenuItem;
        private ButtonStripItem buttonRepeatPlaylist;
        private DataGridViewTextBoxColumn Playing;
        private DataGridViewTextBoxColumn Number;
        private DataGridViewTextBoxColumn FullPath;
        private DataGridViewTextBoxColumn CurrentDirectory;
        private DataGridViewTextBoxColumn Title;
        private DataGridViewTextBoxColumn SkipChapters;
        private DataGridViewTextBoxColumn EndChapter;
        private DataGridViewTextBoxColumn Duration;
    }
}
