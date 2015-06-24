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
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Windows.Forms;
using System.Windows.Forms.Design;

namespace Mpdn.Extensions.PlayerExtensions.Playlist
{
    public partial class PlaylistForm : FormEx
    {
        private Playlist playListUi;

        public static int PlaylistCount { get; set; }

        private const double MaxOpacity = 1.0;
        private const double MinOpacity = 0.8;
        private const string ActiveIndicator = "[*]";
        private const string InactiveIndicator = "[ ]";

        private bool firstShow = true;
        private bool wasShowing;

        private bool columnsFixed = false;

        public List<PlaylistItem> Playlist { get; set; }
        public PlaylistItem CurrentItem { get; set; }

        private int currentPlayIndex = -1;
        private int selectedRowIndex = -1;
        private long previousChapterPosition;

        private bool isDragging;
        private Rectangle dragRowRect;
        private int dragRowIndex;

        private int titleCellIndex = 4;
        private int skipCellIndex = 5;
        private int endCellIndex = 6;

        public Point WindowPosition { get; set; }
        public Size WindowSize { get; set; }
        public bool RememberWindowPosition { get; set; }
        public bool RememberWindowSize { get; set; }
        public bool SnapWithPlayer { get; set; }
        public bool KeepSnapped { get; set; }
        public bool LockWindowSize { get; set; }
        public bool BeginPlaybackOnStartup { get; set; }
        public bool BeginPlaybackWhenFileIsAdded { get; set; }
        public bool BeginPlaybackWhenPlaylistFileIsOpened { get; set; }
        public List<string> Columns { get; set; }
        public List<string> TempRememberedFiles { get; set; }

        public event EventHandler PlaylistChanged;

        public PlaylistForm()
        {
            InitializeComponent();
            Opacity = MinOpacity;
        }

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();

                if (Playlist != null)
                {
                    PlayerControl.PlayerStateChanged -= PlayerStateChanged;
                    PlayerControl.PlaybackCompleted -= PlaybackCompleted;
                    PlayerControl.FrameDecoded -= FrameDecoded;
                    PlayerControl.FramePresented -= FramePresented;
                    PlayerControl.EnteringFullScreenMode -= EnteringFullScreenMode;
                    PlayerControl.ExitedFullScreenMode -= ExitedFullScreenMode;
                }
            }
            base.Dispose(disposing);
        }

        public void Show(Control owner)
        {
            if (PlayerControl.InFullScreenMode)
                return;

            Hide();
            SetLocation(owner);
            timer.Enabled = true;
            dgv_PlayList.Focus();
            base.Show(owner);
        }

        public void Setup(Playlist playListUi)
        {
            if (Playlist != null)
                return;

            this.playListUi = playListUi;
            Icon = PlayerControl.ApplicationIcon;
            DoubleBuffered = true;

            Load += PlaylistForm_Load;
            Shown += PlaylistForm_Shown;
            Resize += PlaylistForm_Resize;

            dgv_PlayList.CellFormatting += dgv_PlayList_CellFormatting;
            dgv_PlayList.CellPainting += dgv_PlayList_CellPainting;
            dgv_PlayList.CellDoubleClick += dgv_PlayList_CellDoubleClick;
            dgv_PlayList.CellEndEdit += dgv_PlayList_CellEndEdit;
            dgv_PlayList.EditingControlShowing += dgv_PlayList_EditingControlShowing;
            dgv_PlayList.MouseMove += dgv_PlayList_MouseMove;
            dgv_PlayList.MouseDown += dgv_PlayList_MouseDown;
            dgv_PlayList.MouseUp += dgv_PlayList_MouseUp;
            dgv_PlayList.KeyDown += dgv_Playlist_KeyDown;
            dgv_PlayList.DragOver += dgv_PlayList_DragOver;
            dgv_PlayList.DragDrop += dgv_PlayList_DragDrop;
            dgv_PlayList.RowsAdded += dgv_PlayList_RowsAdded;
            dgv_PlayList.RowsRemoved += dgv_PlayList_RowsRemoved;
            dgv_PlayList.SelectionChanged += dgv_PlayList_SelectionChanged;

            PlayerControl.PlayerStateChanged += PlayerStateChanged;
            PlayerControl.PlaybackCompleted += PlaybackCompleted;
            PlayerControl.FrameDecoded += FrameDecoded;
            PlayerControl.FramePresented += FramePresented;
            PlayerControl.EnteringFullScreenMode += EnteringFullScreenMode;
            PlayerControl.ExitedFullScreenMode += ExitedFullScreenMode;

            Playlist = new List<PlaylistItem>();
            TempRememberedFiles = new List<string>();
            SetControlStates();
            DisableTabStop(this);
        }

        private void DisableTabStop(Control c)
        {
            if (c.GetType() != typeof(DataGridView))
            {
                c.TabStop = false;

                foreach (Control i in c.Controls)
                {
                    DisableTabStop(i);
                }
            }
        }

        private void SelectNextEditableCell()
        {
            DataGridViewCell currentCell = dgv_PlayList.CurrentCell;
            if (currentCell != null)
            {
                int nextRow = currentCell.RowIndex;

                DataGridViewCell nextCell = SkipChapters.Visible ? dgv_PlayList.Rows[nextRow].Cells[skipCellIndex] : dgv_PlayList.Rows[nextRow].Cells[endCellIndex];
                if (nextCell != null && nextCell.Visible)
                {
                    dgv_PlayList.CurrentCell = nextCell;
                }
            }
        }

        void dgv_PlayList_SelectionChanged(object sender, EventArgs e)
        {
            if (Playlist.Count < 1) return;

            if (dgv_PlayList.SelectedRows.Count > 0)
            {
                openFileDialog.InitialDirectory = Path.GetDirectoryName(Playlist[dgv_PlayList.SelectedRows[0].Index].FilePath);
            }
        }

        void PlaylistForm_Resize(object sender, EventArgs e)
        {
            SetPlaylistToFill();
        }

        void PlaylistForm_Load(object sender, EventArgs e)
        {
            SetColumnSize();
        }

        void PlaylistForm_Shown(object sender, EventArgs e)
        {
            SetColumnSize();
        }

        private void SetColumnSize()
        {
            if (!columnsFixed)
            {
                if (Columns == null || Columns.Count == 0) return;

                for (int i = 0; i < dgv_PlayList.Columns.Count; i++)
                {
                    var c = dgv_PlayList.Columns[i];
                    string[] split = Columns[i].Split('|');
                    if (split[0] == c.Name)
                    {
                        if (split[0] != "Title")
                        {
                            c.Visible = Convert.ToBoolean(split[1]);
                        }

                        c.Width = int.Parse(split[2]);
                        c.FillWeight = int.Parse(split[2]);
                    }
                }

                columnsFixed = true;
            }
        }

        public void ClearPlaylist()
        {
            Playlist.Clear();
            currentPlayIndex = -1;
        }

        public void PopulatePlaylist()
        {
            dgv_PlayList.Rows.Clear();
            if (Playlist.Count == 0) return;

            int fileCount = 1;

            foreach (var i in Playlist)
            {
                string path = Path.GetDirectoryName(i.FilePath);
                string directory = path.Substring(path.LastIndexOf("\\") + 1);
                string file = Path.GetFileName(i.FilePath);

                if (i.SkipChapters != null)
                {
                    if (i.EndChapter != -1)
                    {
                        dgv_PlayList.Rows.Add(new Bitmap(25, 25), fileCount, path, directory, file, String.Join(",", i.SkipChapters),
                            i.EndChapter);
                    }
                    else
                    {
                        dgv_PlayList.Rows.Add(new Bitmap(25, 25), fileCount, path, directory, file, String.Join(",", i.SkipChapters));
                    }
                }
                else
                {
                    dgv_PlayList.Rows.Add(new Bitmap(25, 25), fileCount, path, directory, file);
                }

                if (!File.Exists(i.FilePath))
                {
                    var f = new Font(dgv_PlayList.DefaultCellStyle.Font, FontStyle.Strikeout);
                    dgv_PlayList.Rows[fileCount - 1].DefaultCellStyle.Font = f;
                    dgv_PlayList.Rows[fileCount - 1].DefaultCellStyle.ForeColor = Color.LightGray;
                }

                fileCount++;
            }

            currentPlayIndex = (Playlist.FindIndex(i => i.Active) > -1) ? Playlist.FindIndex(i => i.Active) : -1;

            if (CurrentItem != null && CurrentItem.Active)
            {
                if (File.Exists(CurrentItem.FilePath))
                    SetPlayStyling();
            }

            NotifyPlaylistChanged();
            PlaylistCount = Playlist.Count;
        }

        public void NewPlaylist(bool closeMedia = false)
        {
            ClearPlaylist();
            PopulatePlaylist();
            CurrentItem = null;
            currentPlayIndex = -1;
            Text = "Playlist";
            dgv_PlayList.Invalidate();

            if (closeMedia)
                CloseMedia();
        }

        public void OpenPlaylist()
        {
            openPlaylistDialog.FileName = savePlaylistDialog.FileName;
            if (openPlaylistDialog.ShowDialog(PlayerControl.Form) != DialogResult.OK) return;

            OpenPlaylist(openPlaylistDialog.FileName);
        }

        public void RefreshPlaylist()
        {
            dgv_PlayList.Invalidate();
        }

        public void OpenPlaylist(string fileName)
        {
            ClearPlaylist();

            try
            {
                using (var sr = new StreamReader(fileName))
                {
                    string line;

                    while ((line = sr.ReadLine()) != null)
                    {
                        if (line.Contains("|"))
                        {
                            ParseWithChapters(line);
                        }
                        else
                        {
                            ParseWithoutChapters(line);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show("Invalid or corrupt playlist file.\nAdditional info: " + ex.Message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

            PopulatePlaylist();

            if (!BeginPlaybackWhenPlaylistFileIsOpened) return;
            PlayActive();
        }

        public void SavePlaylist()
        {
            if (Playlist.Count == 0) return;

            savePlaylistDialog.FileName = openPlaylistDialog.FileName;
            if (savePlaylistDialog.ShowDialog(PlayerControl.Form) != DialogResult.OK) return;

            SavePlaylist(savePlaylistDialog.FileName);
        }

        public void SavePlaylist(string filename)
        {
            IEnumerable<string> playlist;
            bool containsChapter = false;

            foreach (var item in Playlist)
            {
                if (item.HasChapter)
                {
                    containsChapter = true;
                }
            }

            if (containsChapter)
            {
                playlist =
                    Playlist
                        .Select(
                            item =>
                                string.Format("{0}{1} | SkipChapter: {2} | EndChapter: {3}",
                                    item.Active ? ActiveIndicator : InactiveIndicator,
                                    item.FilePath, item.HasChapter ? String.Join(",", item.SkipChapters) : "0",
                                    item.EndChapter > -1 ? item.EndChapter : 0));
            }
            else
            {
                playlist =
                    Playlist
                        .Select(
                            item =>
                                string.Format("{0}{1}", item.Active ? ActiveIndicator : InactiveIndicator,
                                    item.FilePath));
            }

            File.WriteAllLines(filename, playlist, Encoding.UTF8);
        }

        public void PlayActive()
        {
            currentPlayIndex = -1;

            foreach (var item in Playlist)
            {
                currentPlayIndex++;
                if (!item.Active) continue;
                OpenMedia();
                return;
            }

            currentPlayIndex = 0;
            OpenMedia();
        }

        public void PlayNext()
        {
            currentPlayIndex++;
            OpenMedia();

            if (currentPlayIndex < Playlist.Count) return;
            currentPlayIndex = Playlist.Count - 1;
        }

        public void PlayPrevious()
        {
            currentPlayIndex--;
            OpenMedia();

            if (currentPlayIndex >= 0) return;
            currentPlayIndex = 0;
        }

        public void PlayNextFileInDirectory(bool next = true)
        {
            if (PlayerControl.PlayerState == PlayerState.Closed)
                return;

            var mediaPath = PlayerControl.MediaFilePath;
            var mediaDir = playListUi.GetDirectoryName(mediaPath);
            var mediaFiles = playListUi.GetMediaFiles(mediaDir);
            var nextFile = next
                ? mediaFiles.SkipWhile(file => file != mediaPath).Skip(1).FirstOrDefault()
                : mediaFiles.TakeWhile(file => file != mediaPath).LastOrDefault();
            if (nextFile != null)
            {
                PlayerControl.OpenMedia(nextFile);
                if (Playlist.Count > 1) AddActiveFile(nextFile);
                else
                {
                    CurrentItem = null;
                    currentPlayIndex = -1;
                    dgv_PlayList.Invalidate();
                }
            }
        }

        public void AddActiveFile(string fileName)
        {
            ResetActive();
            var item = new PlaylistItem(fileName, true) { EndChapter = -1 };
            Playlist.Add(item);
            CurrentItem = item;
            PopulatePlaylist();
            dgv_PlayList.CurrentCell = dgv_PlayList.Rows[currentPlayIndex].Cells[0];

            Text = PlayerControl.PlayerState + " ─ " + CurrentItem.FilePath;
        }

        public void ActiveFile(string fileName)
        {
            ResetActive();
            var item = new PlaylistItem(fileName, true);
            ClearPlaylist();
            Playlist.Add(item);
            CurrentItem = item;
            PopulatePlaylist();

            Text = PlayerControl.PlayerState + " ─ " + CurrentItem.FilePath;
        }

        public void AddFiles(string[] fileNames)
        {
            foreach (var item in fileNames.Select(s => new PlaylistItem(s, false) { EndChapter = -1 }))
            {
                Playlist.Add(item);
            }

            if (dgv_PlayList.CurrentRow != null) selectedRowIndex = dgv_PlayList.CurrentRow.Index;

            PopulatePlaylist();

            if (selectedRowIndex < 0)
            {
                selectedRowIndex = 0;
            }
            else if (selectedRowIndex > Playlist.Count - 1)
            {
                selectedRowIndex = Playlist.Count - 1;
            }

            dgv_PlayList.CurrentCell = dgv_PlayList.Rows[selectedRowIndex].Cells[titleCellIndex];

            if (!BeginPlaybackWhenFileIsAdded) return;

            currentPlayIndex = fileNames.Count() > 1 ? Playlist.Count - fileNames.Count() : currentPlayIndex = Playlist.Count - 1;
            OpenMedia();
        }
        
        public void InsertFile(int index, string fileName)
        {
            PlaylistItem item = new PlaylistItem(fileName, false);
            Playlist.Insert(index, item);
            PopulatePlaylist();
        }

        public void RemoveFile(int index)
        {
            Playlist.RemoveAt(index);
            PopulatePlaylist();
        }

        public void SetPlaylistIndex(int index)
        {
            currentPlayIndex = index;
            OpenMedia();
        }

        private void PlaySelectedFile()
        {
            if (dgv_PlayList.Rows.Count < 1) return;
            if (File.Exists(Playlist[dgv_PlayList.CurrentRow.Index].FilePath))
            {
                var f = new Font(dgv_PlayList.DefaultCellStyle.Font, FontStyle.Regular);
                dgv_PlayList.Rows[dgv_PlayList.CurrentRow.Index].DefaultCellStyle.Font = f;
                dgv_PlayList.Rows[dgv_PlayList.CurrentRow.Index].DefaultCellStyle.ForeColor = Color.Black;
                SetPlaylistIndex(dgv_PlayList.CurrentRow.Index);
            }
            else
            {
                var f = new Font(dgv_PlayList.DefaultCellStyle.Font, FontStyle.Strikeout);
                dgv_PlayList.Rows[dgv_PlayList.CurrentRow.Index].DefaultCellStyle.Font = f;
                dgv_PlayList.Rows[dgv_PlayList.CurrentRow.Index].DefaultCellStyle.ForeColor = Color.LightGray;
            }
        }

        public void CloseMedia()
        {
            CurrentItem = null;
            currentPlayIndex = -1;
            Text = "Playlist";
            PlayerControl.CloseMedia();
            dgv_PlayList.Invalidate();
        }

        private void SetLocation(Control owner)
        {
            int borderWidth = SystemInformation.SizingBorderWidth;

            if (RememberWindowPosition && RememberWindowSize)
            {
                Location = WindowPosition;
                Size = WindowSize;
            }
            else
            {
                if (RememberWindowPosition)
                {
                    if (firstShow)
                    {
                        Location = WindowPosition;
                        firstShow = false;
                    }
                }
                else
                {
                    if (LockWindowSize)
                    {
                        Left = PlayerControl.Form.Right + borderWidth;
                        Top = PlayerControl.Form.Top + borderWidth;

                    }
                    else
                    {
                        Left = PlayerControl.Form.Right;
                        Top = PlayerControl.Form.Top;
                    }
                }
                if (RememberWindowSize)
                {
                    if (firstShow)
                    {
                        Size = WindowSize;
                        firstShow = false;
                    }
                }
                else
                {
                    var mpdnRememberBounds = PlayerControl.PlayerSettings.GeneralSettings.RememberWindowSizePos;
                    var mpdnBounds = PlayerControl.PlayerSettings.GeneralSettings.WindowBounds;

                    var screen = Screen.FromControl(owner);
                    var screenBounds = screen.WorkingArea;

                    if (mpdnRememberBounds)
                        Width = mpdnBounds.Right + mpdnBounds.Width >= (screenBounds.Width / 2) ? screenBounds.Width - (mpdnBounds.Width + mpdnBounds.Left) : Width;
                    else
                        Width = PlayerControl.Form.Right + PlayerControl.Form.Width >= (screenBounds.Width / 2) ? (screenBounds.Width / 2) - PlayerControl.Form.Width / 2 : Width;

                    if (LockWindowSize)
                    {
                        Width = Width - borderWidth;
                        Height = PlayerControl.Form.Height - (borderWidth * 2);
                    }
                    else
                    {
                        Height = PlayerControl.Form.Height;
                    }
                }
            }
            
            if (SnapWithPlayer) { playListUi.SnapPlayer(); }
        }

        private void TimerTick(object sender, EventArgs e)
        {
            HandleOpacity();
        }

        private void HandleOpacity()
        {
            var pos = MousePosition;
            bool inForm = pos.X >= Left && pos.Y >= Top && pos.X < Right && pos.Y < Bottom;

            if (inForm || ActiveForm == this)
            {
                if (Opacity < MaxOpacity)
                {
                    Opacity += 0.1;
                }
            }
            else if (Opacity > MinOpacity)
            {
                Opacity -= 0.1;
            }
        }

        void dgv_PlayList_CellFormatting(object sender, DataGridViewCellFormattingEventArgs e)
        {
            var skipChapterCell = dgv_PlayList.Rows[e.RowIndex].Cells[skipCellIndex];
            var endChapterCell = dgv_PlayList.Rows[e.RowIndex].Cells[endCellIndex];

            if (skipChapterCell.IsInEditMode || endChapterCell.IsInEditMode)
            {
                e.CellStyle.ForeColor = Color.Black;
            }
        }

        void dgv_PlayList_CellPainting(object sender, DataGridViewCellPaintingEventArgs e)
        {
            e.Paint(e.ClipBounds, DataGridViewPaintParts.All);

            bool paintPlayRow = CurrentItem != null && e.RowIndex > -1 && e.RowIndex == currentPlayIndex;
            if (!paintPlayRow) return;

            var brush = new SolidBrush(Color.FromArgb(42, 127, 183));

            if (e.ColumnIndex == 0)
            {
                var rect = new Rectangle(e.CellBounds.X + 15, e.CellBounds.Y + 4, e.CellBounds.Width, e.CellBounds.Height - 9);
                var playIcon = (Bitmap) PlayButton.BackgroundImage;
                var offset = new Point(e.CellBounds.X, e.CellBounds.Y + 2);
                e.Graphics.FillRectangle(brush, rect);
                e.Graphics.DrawImage(playIcon, offset);
            }
            else
            {
                var rect = new Rectangle(e.CellBounds.X, e.CellBounds.Y + 4, e.CellBounds.Width, e.CellBounds.Height - 9);
                e.Graphics.FillRectangle(brush, rect);
            }

            e.Paint(e.ClipBounds, DataGridViewPaintParts.ContentForeground);
            e.Handled = true;
        }

        private void dgv_PlayList_CellDoubleClick(object sender, DataGridViewCellEventArgs e)
        {
            PlaySelectedFile();
        }

        private void dgv_PlayList_CellEndEdit(object sender, DataGridViewCellEventArgs e)
        {
            ParseChapterInput();
        }

        void dgv_PlayList_EditingControlShowing(object sender, DataGridViewEditingControlShowingEventArgs e)
        {
            e.Control.KeyPress -= dgv_PlayList_HandleInput;
            if (dgv_PlayList.CurrentCell.ColumnIndex <= 1) return;

            var tb = e.Control as TextBox;
            if (tb != null)
            {
                tb.KeyPress += dgv_PlayList_HandleInput;
            }
        }

        private void dgv_PlayList_HandleInput(object sender, KeyPressEventArgs e)
        {
            if (!char.IsControl(e.KeyChar) && !char.IsDigit(e.KeyChar) && e.KeyChar != ',' && e.KeyChar != ' ' && dgv_PlayList.CurrentCell.ColumnIndex == skipCellIndex)
            {
                var toolTip = new ToolTip();
                var cell = dgv_PlayList.CurrentCell;
                var cellDisplayRect = dgv_PlayList.GetCellDisplayRectangle(cell.ColumnIndex, cell.RowIndex, false);
                toolTip.Show("Only numbers are allowed. You may separate them with a comma or a space.", dgv_PlayList,
                              cellDisplayRect.X + cell.Size.Width / 2,
                              cellDisplayRect.Y + cell.Size.Height / 2,
                              2000);

                e.Handled = true;
            }

            if (!char.IsControl(e.KeyChar) && !char.IsDigit(e.KeyChar) && dgv_PlayList.CurrentCell.ColumnIndex == endCellIndex)
            {
                var toolTip = new ToolTip();
                var cell = dgv_PlayList.CurrentCell;
                var cellDisplayRect = dgv_PlayList.GetCellDisplayRectangle(cell.ColumnIndex, cell.RowIndex, false);
                toolTip.Show("Only numbers are allowed.", dgv_PlayList,
                              cellDisplayRect.X + cell.Size.Width / 2,
                              cellDisplayRect.Y + cell.Size.Height / 2,
                              2000);

                e.Handled = true;
            }
        }

        void dgv_PlayList_MouseMove(object sender, MouseEventArgs e)
        {
            if (Playlist.Count < 2) return;
            if (e.Button != MouseButtons.Left) return;
            if (dragRowRect != Rectangle.Empty && !dragRowRect.Contains(e.X, e.Y) && isDragging)
            {
                dgv_PlayList.DoDragDrop(dgv_PlayList.Rows[dragRowIndex], DragDropEffects.Move);
            }
        }

        void dgv_PlayList_MouseDown(object sender, MouseEventArgs e)
        {
            var hit = dgv_PlayList.HitTest(e.X, e.Y);
            dragRowIndex = dgv_PlayList.HitTest(e.X, e.Y).RowIndex;

            if (dragRowIndex != -1)
            {
                isDragging = true;
                var dragSize = SystemInformation.DragSize;
                dragRowRect = new Rectangle(new Point(e.X - (dragSize.Width / 2), e.Y - (dragSize.Height / 2)), dragSize);
            }
            else
            {
                dragRowRect = Rectangle.Empty;
                SetPlaylistToFill();
            }

            if (e.Button == MouseButtons.Right)
            {
                if (hit.Type == DataGridViewHitTestType.ColumnHeader)
                {
                    SetColumnStates();
                    dgv_PlaylistColumnContextMenu.Show(Cursor.Position);
                }
                else
                {
                    dgv_PlaylistContextMenu.Show(Cursor.Position);
                }
            }
        }

        void dgv_PlayList_MouseUp(object sender, MouseEventArgs e)
        {
            dragRowIndex = dgv_PlayList.HitTest(e.X, e.Y).RowIndex;

            if (dragRowIndex != -1)
            {
                isDragging = false;
            }
        }

        void dgv_Playlist_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyCode == Keys.Tab)
            {
                SelectNextEditableCell();
                e.SuppressKeyPress = true;
                e.Handled = true;
            }

            if (e.KeyCode == Keys.Delete)
            {
                RemoveSelectedItems();
            }

            if (e.KeyCode == Keys.Enter)
            {
                PlaySelectedFile();
                e.Handled = true;
            }
        }

        void dgv_PlayList_DragOver(object sender, DragEventArgs e)
        {
            e.Effect = DragDropEffects.Move;
        }

        void dgv_PlayList_DragDrop(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                var files = (string[])e.Data.GetData(DataFormats.FileDrop);
                if (files.Length == 1)
                {
                    var filename = files[0];

                    if (Directory.Exists(filename))
                    {
                        var media = playListUi.GetAllMediaFiles(filename);
                        AddFiles(media.ToArray());
                        return;
                    }
                    else if (PlayerExtensions.Playlist.Playlist.IsPlaylistFile(filename))
                    {
                        OpenPlaylist(filename);
                        return;
                    }
                }

                var mediaFiles = new List<string>();

                foreach (var p in files)
                {
                    FileAttributes attr = File.GetAttributes(p);
                    bool isFolder = (attr & FileAttributes.Directory) == FileAttributes.Directory;

                    if (isFolder)
                    {
                        if (Directory.Exists(p))
                        {
                            mediaFiles.AddRange(playListUi.GetAllMediaFiles(p));
                        }
                    }
                }

                if (mediaFiles.Count > 0)
                {
                    AddFiles(mediaFiles.NaturalSort().ToArray());
                }

                var actualFiles =
                      files.Where(file => !Directory.Exists(file))
                          .Where(f => Path.GetExtension(f).Length > 0)
                          .Where(file => openFileDialog.Filter.Contains(Path.GetExtension(file.ToLower())))
                          .OrderBy(f => f, new NaturalSortComparer()).ToList();
                AddFiles(actualFiles.NaturalSort().ToArray());

                dgv_PlayList.CurrentCell = dgv_PlayList.Rows[dgv_PlayList.Rows.Count - 1].Cells[titleCellIndex];
                SetPlayStyling();
            }
            else if (e.Data.GetDataPresent(typeof(DataGridViewRow)))
            {
                var clientPoint = dgv_PlayList.PointToClient(new Point(e.X, e.Y));
                int destinationRow = dgv_PlayList.HitTest(clientPoint.X, clientPoint.Y).RowIndex;

                if (destinationRow == -1 || destinationRow >= Playlist.Count) return;
                var playItem = Playlist.ElementAt(dragRowIndex);
                Playlist.RemoveAt(dragRowIndex);
                NotifyPlaylistChanged();
                Playlist.Insert(destinationRow, playItem);
                PopulatePlaylist();
                dgv_PlayList.CurrentCell = dgv_PlayList.Rows[destinationRow].Cells[titleCellIndex];
            }
        }

        void dgv_PlayList_RowsRemoved(object sender, DataGridViewRowsRemovedEventArgs e)
        {
            SetControlStates();
        }

        private void dgv_PlayList_RowsAdded(object sender, DataGridViewRowsAddedEventArgs e)
        {
            SetControlStates();
        }

        private void SetControlStates()
        {
            if (Playlist.Count > 1)
            {
                buttonLeft.Enabled = true;
                buttonRight.Enabled = true;
                buttonSortAscending.Enabled = true;
                buttonSortDescending.Enabled = true;
                buttonShuffle.Enabled = true;
                buttonRestore.Enabled = true;
                nextItemToolStripMenuItem.Enabled = true;
                previousItemToolStripMenuItem.Enabled = true;
                sortToolStripMenuItem.Enabled = true;
                ascendingToolStripMenuItem.Enabled = true;
                descendingToolStripMenuItem.Enabled = true;
                shuffleToolStripMenuItem.Enabled = true;
                buttonLeft.BackgroundImage = buttonLeftEnabled.BackgroundImage;
                buttonRight.BackgroundImage = buttonRightEnabled.BackgroundImage;
                buttonShuffle.BackgroundImage = buttonShuffleEnabled.BackgroundImage;
                buttonRestore.BackgroundImage = buttonRestoreEnabled.BackgroundImage;
                buttonSortAscending.BackgroundImage = buttonSortAscendingEnabled.BackgroundImage;
                buttonSortDescending.BackgroundImage = buttonSortDescendingEnabled.BackgroundImage;
            }
            else
            {
                buttonLeft.Enabled = false;
                buttonRight.Enabled = false;
                buttonSortAscending.Enabled = false;
                buttonSortDescending.Enabled = false;
                buttonShuffle.Enabled = false;
                buttonRestore.Enabled = false;
                nextItemToolStripMenuItem.Enabled = false;
                previousItemToolStripMenuItem.Enabled = false;
                sortToolStripMenuItem.Enabled = false;
                ascendingToolStripMenuItem.Enabled = false;
                descendingToolStripMenuItem.Enabled = false;
                shuffleToolStripMenuItem.Enabled = false;
                buttonLeft.BackgroundImage = buttonLeftDisabled.BackgroundImage;
                buttonRight.BackgroundImage = buttonRightDisabled.BackgroundImage;
                buttonShuffle.BackgroundImage = buttonShuffleDisabled.BackgroundImage;
                buttonRestore.BackgroundImage = buttonRestoreDisabled.BackgroundImage;
                buttonSortAscending.BackgroundImage = buttonSortAscendingDisabled.BackgroundImage;
                buttonSortDescending.BackgroundImage = buttonSortDescendingDisabled.BackgroundImage;
            }

            if (Playlist.Count > 0)
            {
                buttonNew.Enabled = true;
                buttonSave.Enabled = true;
                buttonDel.Enabled = true;
                newPlaylistToolStripMenuItem.Enabled = true;
                savePlaylistToolStripMenuItem.Enabled = true;
                openFolderToolStripMenuItem.Enabled = true;
                playFileToolStripMenuItem.Enabled = true;
                removeFilesToolStripMenuItem.Enabled = true;
                buttonNew.BackgroundImage = buttonNewEnabled.BackgroundImage;
                buttonSave.BackgroundImage = buttonSaveEnabled.BackgroundImage;
                buttonDel.BackgroundImage = buttonDelEnabled.BackgroundImage;
            }
            else
            {
                buttonNew.Enabled = false;
                buttonSave.Enabled = false;
                buttonDel.Enabled = false;
                newPlaylistToolStripMenuItem.Enabled = false;
                savePlaylistToolStripMenuItem.Enabled = false;
                openFolderToolStripMenuItem.Enabled = false;
                playFileToolStripMenuItem.Enabled = false;
                removeFilesToolStripMenuItem.Enabled = false;
                buttonNew.BackgroundImage = buttonNewDisabled.BackgroundImage;
                buttonSave.BackgroundImage = buttonSaveDisabled.BackgroundImage;
                buttonDel.BackgroundImage = buttonDelDisabled.BackgroundImage;
            }
        }
        private void SetPlaylistToFill()
        {
            foreach (DataGridViewColumn c in dgv_PlayList.Columns)
            {
                if (c.Name != "Playing")
                {
                    c.AutoSizeMode = DataGridViewAutoSizeColumnMode.Fill;
                }
            }
        }

        private void SetColumnStates()
        {
            numberToolStripMenuItem.Checked = Number.Visible;
            directoryToolStripMenuItem.Checked = CurrentDirectory.Visible;
            fullPathToolStripMenuItem.Checked = FullPath.Visible;
            skipChaptersToolStripMenuItem.Checked = SkipChapters.Visible;
            endChapterToolStripMenuItem.Checked = EndChapter.Visible;

            titleCellIndex = Title.Index;
            skipCellIndex = SkipChapters.Index;
            endCellIndex = EndChapter.Index;
        }

        private void PlayerStateChanged(object sender, PlayerStateEventArgs e)
        {
            if (CurrentItem == null) return;
            Text = PlayerControl.PlayerState + " - " + CurrentItem.FilePath;
        }

        private void PlaybackCompleted(object sender, EventArgs e)
        {
            if (PlayerControl.PlayerState == PlayerState.Closed) return;
            if (PlayerControl.MediaPosition == PlayerControl.MediaDuration)
            {
                PlayNext();
            }
        }

        private void EnteringFullScreenMode(object sender, EventArgs e)
        {
            wasShowing = Visible;
            Hide();
        }

        private void ExitedFullScreenMode(object sender, EventArgs e)
        {
            if (wasShowing)
            {
                Show(PlayerControl.VideoPanel);
            }
        }

        private void PlaylistFormClosing(object sender, FormClosingEventArgs e)
        {
            if (e.CloseReason != CloseReason.UserClosing) return;
            e.Cancel = true;
            Hide();
            timer.Enabled = false;
        }

        private void FrameDecoded(object sender, FrameEventArgs e)
        {
            if (PlayerControl.MediaFilePath != "" && PlayerControl.Chapters.Count != 0 && CurrentItem != null && CurrentItem.HasChapter)
            {
                previousChapterPosition = GetChapters().Aggregate((prev, next) => e.SampleTime >= prev.Position && e.SampleTime <= next.Position ? prev : next).Position;
            }
        }

        private void FramePresented(object sender, FrameEventArgs e)
        {
            if (PlayerControl.MediaFilePath != "" && PlayerControl.Chapters.Count != 0 && CurrentItem != null && CurrentItem.HasChapter)
            {
                if (e.SampleTime >= previousChapterPosition)
                {
                    int currentChapterIndex = GetChapterIndexByPosition(previousChapterPosition);

                    if (CurrentItem.SkipChapters.Contains(currentChapterIndex) && currentChapterIndex < PlayerControl.Chapters.Count)
                    {
                        SelectChapter(currentChapterIndex);
                    }
                    else if (currentChapterIndex == CurrentItem.EndChapter)
                    {
                        PlayNext();
                    }
                }
            }
        }

        private void ParseChapterInput()
        {
            try
            {
                for (int i = 0; i < dgv_PlayList.Rows.Count; i++)
                {
                    var skipChapterCell = dgv_PlayList.Rows[i].Cells[skipCellIndex];
                    var endChapterCell = dgv_PlayList.Rows[i].Cells[endCellIndex];

                    if (skipChapterCell.Value != null && skipChapterCell.Value.ToString() != "")
                    {
                        var formattedValue = Regex.Replace(skipChapterCell.Value.ToString(), @"[^0-9,\s]*", "");
                        var numbers = formattedValue.Trim().Replace(" ", ",").Split(',');
                        var sortedNumbers = numbers.Distinct().Except(new[] { "" }).Select(int.Parse).OrderBy(x => x).ToList();

                        if (CurrentItem != null && i == currentPlayIndex)
                        {
                            if (sortedNumbers.Any(num => num >= PlayerControl.Chapters.Count))
                            {
                                var toolTip = new ToolTip();
                                var cellDisplayRect = dgv_PlayList.GetCellDisplayRectangle(skipChapterCell.ColumnIndex,
                                    skipChapterCell.RowIndex, false);
                                toolTip.Show("Only numbers < " + PlayerControl.Chapters.Count + " are allowed",
                                    dgv_PlayList,
                                    cellDisplayRect.X + skipChapterCell.Size.Width / 2,
                                    cellDisplayRect.Y + skipChapterCell.Size.Height / 2,
                                    2000);
                                sortedNumbers.RemoveAll(num => num >= PlayerControl.Chapters.Count);
                            }
                            if (PlayerControl.Chapters.Count == 0)
                            {
                                sortedNumbers.Clear();
                            }
                        }

                        formattedValue = String.Join(",", sortedNumbers);
                        skipChapterCell.Value = formattedValue;
                    }

                    if (endChapterCell.Value != null && endChapterCell.Value.ToString() != "")
                    {
                        var value = new String(endChapterCell.Value.ToString().Where(Char.IsDigit).ToArray());

                        if (CurrentItem != null && i == currentPlayIndex)
                        {
                            if (value.Length > 0 && int.Parse(value) > PlayerControl.Chapters.Count)
                            {
                                var toolTip = new ToolTip();
                                var cellDisplayRect = dgv_PlayList.GetCellDisplayRectangle(endChapterCell.ColumnIndex,
                                    endChapterCell.RowIndex, false);
                                toolTip.Show("Only numbers <= " + PlayerControl.Chapters.Count + " are allowed",
                                    dgv_PlayList,
                                    cellDisplayRect.X + endChapterCell.Size.Width / 2,
                                    cellDisplayRect.Y + endChapterCell.Size.Height / 2,
                                    2000);

                                value = PlayerControl.Chapters.Count.ToString();
                            }
                            if (PlayerControl.Chapters.Count == 0)
                            {
                                value = "";
                            }
                        }

                        endChapterCell.Value = value;
                    }
                }

                UpdatePlaylist();
            }
            catch (Exception ex)
            {
                PlayerControl.HandleException(ex);
            }
        }

        private void UpdatePlaylist()
        {
            try
            {
                for (int i = 0; i < dgv_PlayList.Rows.Count; i++)
                {
                    var skipChapters = new List<int>();
                    int endChapter = -1;

                    var skipChapterCell = dgv_PlayList.Rows[i].Cells[skipCellIndex];
                    var endChapterCell = dgv_PlayList.Rows[i].Cells[endCellIndex];

                    if (skipChapterCell.Value != null && skipChapterCell.Value.ToString() != "")
                    {
                        skipChapters = skipChapterCell.Value.ToString().Split(',').Select(int.Parse).ToList();
                        Playlist.ElementAt(i).HasChapter = true;
                    }

                    if (endChapterCell.Value != null && endChapterCell.Value.ToString() != "")
                    {
                        endChapter = int.Parse(endChapterCell.Value.ToString());
                    }

                    Playlist.ElementAt(i).SkipChapters = skipChapters;
                    Playlist.ElementAt(i).EndChapter = endChapter;
                }
            }
            catch (Exception ex)
            {
                PlayerControl.HandleException(ex);
            }
        }

        private void ParseWithoutChapters(string line)
        {
            string title = "";
            bool isActive = false;

            if (line.StartsWith(ActiveIndicator))
            {
                title = line.Substring(ActiveIndicator.Length).Trim();
                isActive = true;
            }
            else if (line.StartsWith(InactiveIndicator))
            {
                title = line.Substring(InactiveIndicator.Length).Trim();
            }
            else
            {
                throw new FileLoadException();
            }

            Playlist.Add(new PlaylistItem(title, isActive));
        }

        private void ParseWithChapters(string line)
        {
            var splitLine = line.Split('|');
            string title = "";
            bool isActive = false;
            var skipChapters = new List<int>();

            if (splitLine[0].StartsWith(ActiveIndicator))
            {
                title = splitLine[0].Substring(ActiveIndicator.Length).Trim();
                isActive = true;
            }
            else if (line.StartsWith(InactiveIndicator))
            {
                title = splitLine[0].Substring(InactiveIndicator.Length).Trim();
                isActive = false;
            }
            else
            {
                throw new FileLoadException();
            }

            if (splitLine[1].Length > 0)
            {
                splitLine[1] = splitLine[1].Substring(splitLine[1].IndexOf(':') + 1).Trim();
                skipChapters = new List<int>(splitLine[1].Split(',').Select(int.Parse));
            }

            var endChapter = int.Parse(splitLine[2].Substring(splitLine[2].IndexOf(':') + 1).Trim());
            Playlist.Add(new PlaylistItem(title, skipChapters, endChapter, isActive));
        }

        private void OpenMedia()
        {
            if (currentPlayIndex < 0 || currentPlayIndex >= Playlist.Count) return;

            bool playerWasFullScreen = PlayerControl.InFullScreenMode;
            ResetActive();

            try
            {
                var item = Playlist[currentPlayIndex];
                dgv_PlayList.CurrentCell = dgv_PlayList.Rows[currentPlayIndex].Cells[titleCellIndex];

                if (File.Exists(item.FilePath))
                {
                    SetPlayStyling();
                    PlayerControl.OpenMedia(item.FilePath);
                }
                else
                {
                    if (currentPlayIndex != Playlist.Count - 1) PlayNext();
                    else CloseMedia();

                    SetPlayStyling();
                    return;
                }

                if (playerWasFullScreen)
                {
                    PlayerControl.GoFullScreen();
                }

                item.Active = true;
                CurrentItem = item;
                previousChapterPosition = 0;

                Text = PlayerControl.PlayerState + " ─ " + CurrentItem.FilePath;
                ParseChapterInput();
            }
            catch (Exception ex)
            {
                PlayerControl.HandleException(ex);
                PlayNext();
            }

            dgv_PlayList.Invalidate();
        }

        private void SortPlayList(bool ascending = true)
        {
            RememberPlaylist();

            if (ascending)
            {
                Playlist = Playlist.OrderBy(f => Path.GetDirectoryName(f.FilePath), new NaturalSortComparer())
                            .ThenBy(f => Path.GetFileName(f.FilePath), new NaturalSortComparer()).ToList();
            }
            else
            {
                Playlist = Playlist.OrderByDescending(f => Path.GetDirectoryName(f.FilePath), new NaturalSortComparer())
                            .ThenBy(f => Path.GetFileName(f.FilePath), new NaturalSortComparer()).ToList();
            }

            PopulatePlaylist();
        }

        private void ShufflePlayList()
        {
            RememberPlaylist();
            Playlist.Shuffle();
            PopulatePlaylist();
        }

        public void SetPlayStyling()
        {
            foreach (DataGridViewRow r in dgv_PlayList.Rows)
            {
                if (File.Exists(Playlist[r.Index].FilePath))
                {
                    var f = new Font(dgv_PlayList.DefaultCellStyle.Font, FontStyle.Regular);
                    r.DefaultCellStyle.Font = f;
                    r.DefaultCellStyle.ForeColor = Color.Black;
                    r.Selected = false;
                }
                else
                {
                    var f = new Font(dgv_PlayList.DefaultCellStyle.Font, FontStyle.Strikeout);
                    r.DefaultCellStyle.Font = f;
                    r.DefaultCellStyle.ForeColor = Color.LightGray;
                }
            }

            if (currentPlayIndex != -1)
            {
                dgv_PlayList.Rows[currentPlayIndex].DefaultCellStyle.ForeColor = Color.White;
                dgv_PlayList.Rows[currentPlayIndex].Selected = true;
            }
        }

        private void SelectChapter(int chapterNum)
        {
            if (PlayerControl.PlayerState == PlayerState.Closed) return;

            var chapters = GetChapters();

            if (chapters.ElementAt(chapterNum) == null) return;
            PlayerControl.SeekMedia(chapters.ElementAt(chapterNum).Position);
            PlayerControl.ShowOsdText(chapters.ElementAt(chapterNum).Name);
        }

        private int GetChapterIndexByPosition(long position)
        {
            int currentChapterIndex = 0;

            foreach (var c in GetChapters().Where(c => c != null))
            {
                currentChapterIndex++;
                if (c.Position != position) continue;
                return currentChapterIndex;
            }

            return 0;
        }

        private IEnumerable<Chapter> GetChapters()
        {
            return PlayerControl.Chapters.OrderBy(chapter => chapter.Position);
        }

        public void FocusPlaylistItem(int index)
        {
            dgv_PlayList.CurrentCell = dgv_PlayList.Rows[index].Cells[titleCellIndex];
        }

        public void FocusPlaylist()
        {
            dgv_PlayList.Focus();
        }

        private void ResetActive()
        {
            foreach (var item in Playlist)
            {
                item.Active = false;
            }
        }

        private void NotifyPlaylistChanged()
        {
            if (PlaylistChanged != null)
                PlaylistChanged(this, null);
        }

        private void AddFilesToPlaylist()
        {
            if (openFileDialog.ShowDialog(this) != DialogResult.OK)
                return;

            var fileNames = openFileDialog.FileNames;
            AddFiles(fileNames);
        }

        private void AddFolderToPlaylist()
        {
            using (Ookii.Dialogs.VistaFolderBrowserDialog fd = new Ookii.Dialogs.VistaFolderBrowserDialog())
            {
                fd.Description = "Add folder";
                fd.UseDescriptionForTitle = true;
                fd.ShowNewFolderButton = true;

                if (fd.ShowDialog(this) != DialogResult.OK)
                    return;

                var media = playListUi.GetAllMediaFiles(fd.SelectedPath);
                if (media.ToArray().Length == 0)
                {
                    MessageBox.Show("There are no files in the selected directory.", "Warning", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    return;
                }

                AddFiles(media.ToArray());
            }
        }

        private void RemoveSelectedItems()
        {
            var rowIndexes = new List<int>();

            try
            {
                if (Playlist.Count <= 0) return;
                if (dgv_PlayList.CurrentRow != null) selectedRowIndex = dgv_PlayList.CurrentRow.Index;

                rowIndexes.AddRange(from DataGridViewRow r in dgv_PlayList.SelectedRows select r.Index);

                foreach (int index in rowIndexes.OrderByDescending(v => v))
                {
                    if (index == currentPlayIndex)
                    {
                        CloseMedia();
                    }

                    Playlist.RemoveAt(index);
                }

                PopulatePlaylist();

                if (selectedRowIndex < 0)
                {
                    selectedRowIndex = 0;
                }
                else if (selectedRowIndex > Playlist.Count - 1)
                {
                    selectedRowIndex = Playlist.Count - 1;
                }

                dgv_PlayList.CurrentCell = Playlist.Count > 0 ? dgv_PlayList.Rows[selectedRowIndex].Cells[titleCellIndex] : dgv_PlayList.CurrentCell = null;
            }
            catch (Exception ex)
            {
                PlayerControl.HandleException(ex);
            }
        }

        public DataGridView GetDgvPlaylist()
        {
            return dgv_PlayList;
        }

        public void RememberPlaylist()
        {
            TempRememberedFiles.Clear();

            if (Playlist.Count > 0)
            {
                foreach (PlaylistItem i in Playlist)
                {
                    string skipChapters = "";

                    if (i.SkipChapters != null && i.SkipChapters.Count > 0)
                    {
                        skipChapters = string.Join(",", i.SkipChapters);
                    }

                    TempRememberedFiles.Add(i.FilePath + "|" + skipChapters + "|" + i.EndChapter + "|" +
                                                 i.Active);
                }
            }
        }

        public void RestoreRememberedPlaylist()
        {
            List<PlaylistItem> playList = new List<PlaylistItem>();

            foreach (var f in TempRememberedFiles)
            {
                string[] s = f.Split('|');
                string filePath = s[0];
                List<int> skipChapters = new List<int>();
                if (s[1].Length > 0)
                {
                    if (s[1].Contains(","))
                    {
                        skipChapters = s[1].Split(',').Select(int.Parse).ToList();
                    }
                    else
                    {
                        skipChapters.Add(int.Parse(s[1]));
                    }
                }
                int endChapter = int.Parse(s[2]);
                bool active = Boolean.Parse(s[3]);

                playList.Add(new PlaylistItem(filePath, skipChapters, endChapter, active));
            }

            Playlist = playList;
            PopulatePlaylist();
            RefreshPlaylist();
        }

        private void OpenContainingFolder()
        {
            if (Playlist.Count == 0) return;
            System.Diagnostics.Process.Start(Path.GetDirectoryName(Playlist[dgv_PlayList.CurrentRow.Index].FilePath));
        }

        private void ButtonAddClick(object sender, EventArgs e)
        {
            AddFilesToPlaylist();
            dgv_PlayList.Focus();
        }

        private void ButtonAddFolderClick(object sender, EventArgs e)
        {
            AddFolderToPlaylist();
            dgv_PlayList.Focus();
        }

        private void ButtonOpenFolderClick(object sender, EventArgs e)
        {
            OpenContainingFolder();
        }

        private void ButtonDelClick(object sender, EventArgs e)
        {
            RemoveSelectedItems();
            dgv_PlayList.Focus();
        }

        private void ButtonNewClick(object sender, EventArgs e)
        {
            NewPlaylist(true);
        }

        private void ButtonOpenClick(object sender, EventArgs e)
        {
            OpenPlaylist();
        }

        private void ButtonSaveClick(object sender, EventArgs e)
        {
            SavePlaylist();
        }

        private void ButtonLeftClick(object sender, EventArgs e)
        {
            PlayPrevious();
        }

        private void ButtonRightClick(object sender, EventArgs e)
        {
            PlayNext();
        }

        private void ButtonSortAscendingClick(object sender, EventArgs e)
        {
            SortPlayList();
        }

        private void ButtonSortDescendingClick(object sender, EventArgs e)
        {
            SortPlayList(false);
        }

        private void ButtonRestoreClick(object sender, EventArgs e)
        {
            RestoreRememberedPlaylist();
        }

        private void ButtonSettingsClick(object sender, EventArgs e)
        {
            playListUi.ShowConfigDialog(this);
            playListUi.Reinitialize();
        }

        private void ButtonPlayClick(object sender, EventArgs e)
        {
            PlaySelectedFile();
        }

        private void ButtonShuffleClick(object sender, EventArgs e)
        {
            ShufflePlayList();
        }

        private void FormKeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyData == Keys.Escape)
            {
                Hide();
            }

            if (e.KeyCode == Keys.P && e.Modifiers == (Keys.Control | Keys.Alt))
            {
                playListUi.ViewPlaylist();
                e.SuppressKeyPress = true;
                e.Handled = true;
            }

            if (e.KeyCode == Keys.Tab && e.Modifiers == Keys.Control)
            {
                if (!PlayerControl.InFullScreenMode && !PlayerControl.Form.ContainsFocus)
                {
                    PlayerControl.Form.Activate();
                    Cursor.Position = new Point(PlayerControl.Form.Location.X + 100, PlayerControl.Form.Location.Y + 100);
                    e.SuppressKeyPress = true;
                    e.Handled = true;
                }
            }
        }

        protected override void WndProc(ref Message message)
        {
            const int WM_SYSCOMMAND = 0x0112;
            const int SC_MOVE = 0xF010;

            switch (message.Msg)
            {
                case WM_SYSCOMMAND:
                    int command = message.WParam.ToInt32() & 0xfff0;
                    if (KeepSnapped && command == SC_MOVE)
                        return;
                    break;
            }

            base.WndProc(ref message);
        }

        private void UpdateColumns(object sender, EventArgs e)
        {
            Number.Visible = numberToolStripMenuItem.Checked;
            FullPath.Visible = fullPathToolStripMenuItem.Checked;
            CurrentDirectory.Visible = directoryToolStripMenuItem.Checked;
            SkipChapters.Visible = skipChaptersToolStripMenuItem.Checked;
            EndChapter.Visible = endChapterToolStripMenuItem.Checked;

            titleCellIndex = Title.Index;
            skipCellIndex = SkipChapters.Index;
            endCellIndex = EndChapter.Index;
        }
    }

    public class PlaylistItem
    {
        public string FilePath { get; set; }
        public bool Active { get; set; }
        public bool HasChapter { get; set; }
        public List<int> SkipChapters { get; set; }
        public int EndChapter { get; set; }

        public PlaylistItem(string filePath, bool isActive)
        {
            if (string.IsNullOrWhiteSpace(filePath))
            {
                throw new ArgumentNullException("filePath");
            }

            FilePath = filePath;
            Active = isActive;
        }

        public PlaylistItem(string filePath, List<int> skipChapter, int endChapter, bool isActive)
        {
            if (string.IsNullOrWhiteSpace(filePath))
            {
                throw new ArgumentNullException("filePath");
            }

            FilePath = filePath;
            Active = isActive;
            SkipChapters = skipChapter;
            EndChapter = endChapter;
            HasChapter = true;
        }

        public override string ToString()
        {
            if (HasChapter)
            {
                return Path.GetFileName(FilePath) + " | SkipChapter: " + String.Join(",", SkipChapters) + " | EndChapter: " + EndChapter ?? "???";
            }

            return Path.GetFileName(FilePath) ?? "???";
        }
    }

    [ToolStripItemDesignerAvailability(ToolStripItemDesignerAvailability.StatusStrip)]
    public class ButtonStripItem : ToolStripControlHostProxy
    {
        public ButtonStripItem()
            : base(CreateButtonInstance())
        {
        }

        private static Button CreateButtonInstance()
        {
            var b = new Button {BackColor = Color.Transparent, FlatStyle = FlatStyle.Flat};
            b.FlatAppearance.BorderSize = 0;
            b.FlatAppearance.BorderColor = Color.FromArgb(0, 255, 255, 255);
            return b;
        }
    }

    public class ToolStripControlHostProxy : ToolStripControlHost
    {
        public ToolStripControlHostProxy()
            : base(new Control())
        {
        }

        public ToolStripControlHostProxy(Control c)
            : base(c)
        {
        }
    }

    public class FormEx : Form
    {
        private float m_ScaleFactorHeight = -1f;
        private float m_ScaleFactorWidth = -1f;

        protected SizeF ScaleFactor { get; private set; }

        protected override void ScaleControl(SizeF factor, BoundsSpecified specified)
        {
            base.ScaleControl(factor, specified);

            if (!(m_ScaleFactorWidth < 0 || m_ScaleFactorHeight < 0))
                return;

            if (m_ScaleFactorWidth < 0 && specified.HasFlag(BoundsSpecified.Width))
            {
                m_ScaleFactorWidth = factor.Width;
            }
            if (m_ScaleFactorHeight < 0 && specified.HasFlag(BoundsSpecified.Height))
            {
                m_ScaleFactorHeight = factor.Height;
            }

            if (m_ScaleFactorWidth < 0 || m_ScaleFactorHeight < 0)
                return;

            ScaleFactor = new SizeF(m_ScaleFactorWidth, m_ScaleFactorHeight);
        }
    }

    [SuppressUnmanagedCodeSecurity]
    internal static class SafeNativeMethods
    {
        [DllImport("shlwapi.dll", CharSet = CharSet.Unicode)]
        public static extern int StrCmpLogicalW(string psz1, string psz2);
    }

    public class NaturalSortComparer : IComparer<string>
    {
        public NaturalSortComparer() : this(false) { }

        public NaturalSortComparer(bool descending)
        {
        }

        public int Compare(string a, string b)
        {
            string[] arrayA = a.Split(Path.DirectorySeparatorChar);
            string[] arrayB = b.Split(Path.DirectorySeparatorChar);

            int length = Math.Max(arrayA.Length, arrayB.Length);

            for (int i = 0; i < length; i++)
            {
                int result = SafeNativeMethods.StrCmpLogicalW(arrayA.Length > i ? arrayA[i].ToLower() : string.Empty, arrayB.Length > i ? arrayB[i].ToLower() : string.Empty);

                if (result != 0)
                    return result;
            }

            return 0;
        }
    }

    public static class ThreadSafeRandom
    {
        [ThreadStatic]
        private static Random Local;

        public static Random ThisThreadsRandom
        {
            get { return Local ?? (Local = new Random(unchecked(Environment.TickCount * 31 + Thread.CurrentThread.ManagedThreadId))); }
        }
    }

    public static class ListExtensions
    {
        public static IList<string> NaturalSort(this IList<string> list)
        {
            return list.OrderBy(f => Path.GetDirectoryName(f), new NaturalSortComparer())
                            .ThenBy(f => Path.GetFileName(f), new NaturalSortComparer()).ToList();
        }

        public static void Shuffle<T>(this IList<T> list)
        {
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = ThreadSafeRandom.ThisThreadsRandom.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }
    }
}
