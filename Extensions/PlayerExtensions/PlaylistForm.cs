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
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows.Forms.Design;
using MediaInfoDotNet;
using Mpdn.Extensions.Framework;
using Ookii.Dialogs;

namespace Mpdn.Extensions.PlayerExtensions.Playlist
{
    public partial class PlaylistForm : FormEx
    {
        #region EventHandlers

        public event EventHandler PlaylistChanged;

        #endregion

        #region Eventhandler Methods

        private void NotifyPlaylistChanged()
        {
            PlaylistChanged.Handle(h => h(this, EventArgs.Empty));
        }

        #endregion

        #region Fields

        private Playlist playListUi;

        private const double MaxOpacity = 1.0;
        private const double MinOpacity = 0.8;
        private const string ActiveIndicator = "[*]";
        private const string InactiveIndicator = "[ ]";

        private string loadedPlaylist;

        private bool firstShow = true;
        private bool wasShowing;

        private bool columnsFixed;

        private int currentPlayIndex = -1;
        private int selectedRowIndex = -1;
        private long previousChapterPosition;

        private bool isDragging;
        private Rectangle dragRowRect;
        private int dragRowIndex;

        private int titleCellIndex = 4;
        private int skipCellIndex = 5;
        private int endCellIndex = 6;

        private int minWorker;
        private int minIoc;

        #endregion

        #region Properties

        public List<PlaylistItem> Playlist { get; set; }
        public PlaylistItem CurrentItem { get; set; }
        public static int PlaylistCount { get; set; }

        public Point WindowPosition { get; set; }
        public Size WindowSize { get; set; }
        public bool RememberWindowPosition { get; set; }
        public bool RememberWindowSize { get; set; }
        public bool SnapWithPlayer { get; set; }
        public bool KeepSnapped { get; set; }
        public bool LockWindowSize { get; set; }
        public bool BeginPlaybackOnStartup { get; set; }
        public List<string> Columns { get; set; }
        public List<string> TempRememberedFiles { get; set; }

        #endregion

        #region PlaylistForm init and dispose

        public PlaylistForm()
        {
            InitializeComponent();
            Opacity = MinOpacity;
        }

        public void Setup(Playlist playListUi)
        {
            if (Playlist != null) return;

            this.playListUi = playListUi;
            Icon = Gui.Icon;
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

            Player.StateChanged += PlayerStateChanged;
            Player.Playback.Completed += PlaybackCompleted;
            Media.Frame.Decoded += FrameDecoded;
            Media.Frame.Presented += FramePresented;
            Player.FullScreenMode.Entering += EnteringFullScreenMode;
            Player.FullScreenMode.Exited += ExitedFullScreenMode;

            Playlist = new List<PlaylistItem>();
            TempRememberedFiles = new List<string>();

            ThreadPool.GetMaxThreads(out minWorker, out minIoc);
            ThreadPool.SetMinThreads(minWorker, minIoc);

            SetControlStates();
            DisableTabStop(this);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();

                if (Playlist != null)
                {
                    Player.StateChanged -= PlayerStateChanged;
                    Player.Playback.Completed -= PlaybackCompleted;
                    Media.Frame.Decoded -= FrameDecoded;
                    Media.Frame.Presented -= FramePresented;
                    Player.FullScreenMode.Entering -= EnteringFullScreenMode;
                    Player.FullScreenMode.Exited -= ExitedFullScreenMode;
                }
            }

            base.Dispose(disposing);
        }

        protected override void WndProc(ref Message message)
        {
            const int WM_SYSCOMMAND = 0x0112;
            const int SC_MOVE = 0xF010;

            switch (message.Msg)
            {
                case WM_SYSCOMMAND:
                    int command = message.WParam.ToInt32() & 0xfff0;
                    if (KeepSnapped && command == SC_MOVE) return;
                    break;
            }

            base.WndProc(ref message);
        }

        public void Show(Control owner)
        {
            if (Player.FullScreenMode.Active) return;

            Hide();
            SetLocation(owner);
            timer.Enabled = true;
            dgv_PlayList.Focus();
            base.Show(owner);
        }

        private void SetLocation(Control owner)
        {
            int borderWidth = SystemInformation.SizingBorderWidth;
            int right = Player.ActiveForm.Right;
            int top = Player.ActiveForm.Top;
            int width = Player.ActiveForm.Width;
            int height = Player.ActiveForm.Height;

            if (RememberWindowPosition && RememberWindowSize)
            {
                if (firstShow)
                {
                    Location = WindowPosition;
                    Size = WindowSize;
                    firstShow = false;
                }
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
                        Left = right + borderWidth;
                        Top = top + borderWidth;
                    }
                    else
                    {
                        Left = right;
                        Top = top;
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
                    bool mpdnRememberBounds = Player.Config.Settings.GeneralSettings.RememberWindowSizePos;
                    var mpdnBounds = Player.Config.Settings.GeneralSettings.WindowBounds;

                    var screen = Screen.FromControl(owner);
                    var screenBounds = screen.WorkingArea;

                    if (mpdnRememberBounds)
                    {
                        Width = mpdnBounds.Right + mpdnBounds.Width >= (screenBounds.Width / 2)
                            ? screenBounds.Width - (mpdnBounds.Width + mpdnBounds.Left)
                            : Width;
                    }
                    else Width = right + width >= (screenBounds.Width / 2) ? (screenBounds.Width / 2) - width / 2 : Width;

                    if (LockWindowSize)
                    {
                        Width = Width - borderWidth;
                        Height = height - (borderWidth * 2);
                    }
                    else Height = height;
                }
            }

            if (SnapWithPlayer) playListUi.SnapPlayer();
        }

        public DataGridView GetDgvPlaylist()
        {
            return dgv_PlayList;
        }

        #endregion

        #region Playlist Methods

        public void PopulatePlaylist()
        {
            dgv_PlayList.Rows.Clear();
            if (Playlist.Count == 0) return;

            var fileCount = 1;

            foreach (var i in Playlist)
            {
                string path = Path.GetDirectoryName(i.FilePath);
                string directory = path.Substring(path.LastIndexOf("\\") + 1);
                string file = Path.GetFileName(i.FilePath);

                if (i.SkipChapters != null)
                {
                    if (i.EndChapter != -1)
                    {
                        dgv_PlayList.Rows.Add(new Bitmap(25, 25), fileCount, path, directory, file,
                            String.Join(",", i.SkipChapters),
                            i.EndChapter, i.Duration);
                    }
                    else
                    {
                        dgv_PlayList.Rows.Add(new Bitmap(25, 25), fileCount, path, directory, file,
                            String.Join(",", i.SkipChapters), null, i.Duration);
                    }
                }
                else dgv_PlayList.Rows.Add(new Bitmap(25, 25), fileCount, path, directory, file, null, null, i.Duration);

                if (!File.Exists(i.FilePath))
                {
                    var f = new Font(dgv_PlayList.DefaultCellStyle.Font, FontStyle.Strikeout);
                    dgv_PlayList.Rows[fileCount - 1].DefaultCellStyle.Font = f;
                    dgv_PlayList.Rows[fileCount - 1].DefaultCellStyle.ForeColor = Color.LightGray;
                }

                fileCount++;
            }

            currentPlayIndex = (Playlist.FindIndex(i => i.Active) > -1) ? Playlist.FindIndex(i => i.Active) : -1;

            if (CurrentItem != null && CurrentItem.Active) if (File.Exists(CurrentItem.FilePath)) SetPlayStyling();

            NotifyPlaylistChanged();
            PlaylistCount = Playlist.Count;
        }

        public void RefreshPlaylist()
        {
            dgv_PlayList.Invalidate();
        }

        public void NewPlaylist(bool closeMedia = false)
        {
            ClearPlaylist();
            PopulatePlaylist();
            CurrentItem = null;
            currentPlayIndex = -1;
            Text = "Playlist";
            dgv_PlayList.Invalidate();

            if (closeMedia) CloseMedia();
        }

        public void ClearPlaylist()
        {
            Playlist.Clear();
            currentPlayIndex = -1;
            playToolStripMenuItem.Text = "Play";
        }

        public void OpenPlaylist(bool clear = true)
        {
            openPlaylistDialog.FileName = savePlaylistDialog.FileName;
            if (openPlaylistDialog.ShowDialog(Player.ActiveForm) != DialogResult.OK) return;

            loadedPlaylist = openPlaylistDialog.FileName;
            OpenPlaylist(openPlaylistDialog.FileName, clear);
        }

        public void OpenPlaylist(string fileName, bool clear = true)
        {
            if (clear) ClearPlaylist();

            try
            {
                using (var sr = new StreamReader(fileName))
                {
                    string line;

                    while ((line = sr.ReadLine()) != null)
                    {
                        if (line.Contains("|")) ParseWithChapters(line);
                        else ParseWithoutChapters(line);
                    }
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show("Invalid or corrupt playlist file.\nAdditional info: " + ex.Message, "Error",
                    MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

            PopulatePlaylist();

            if (clear) PlayActive();
        }

        private void SavePlaylist()
        {
            if (String.IsNullOrEmpty(loadedPlaylist)) return;
            SavePlaylist(loadedPlaylist);
        }

        public void SavePlaylist(string filename)
        {
            IEnumerable<string> playlist;
            var containsChapter = false;

            foreach (var item in Playlist.Where(item => item.HasChapter))
            {
                containsChapter = true;
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

            var t = new ToolTip();
            t.Show("Playlist saved!", this, PointToClient(Cursor.Position), 2000);
        }

        public void SavePlaylistAs()
        {
            if (Playlist.Count == 0) return;

            savePlaylistDialog.FileName = openPlaylistDialog.FileName;
            if (savePlaylistDialog.ShowDialog(Player.ActiveForm) != DialogResult.OK) return;

            SavePlaylist(savePlaylistDialog.FileName);
        }

        #endregion

        #region Parsing Methods

        private void ParseChapterInput()
        {
            if (!SkipChapters.Visible && !EndChapter.Visible) return;

            try
            {
                for (var i = 0; i < dgv_PlayList.Rows.Count; i++)
                {
                    var skipChapterCell = dgv_PlayList.Rows[i].Cells[skipCellIndex];
                    var endChapterCell = dgv_PlayList.Rows[i].Cells[endCellIndex];

                    if (skipChapterCell.Value != null && skipChapterCell.Value.ToString() != string.Empty)
                    {
                        string formattedValue = Regex.Replace(skipChapterCell.Value.ToString(), @"[^0-9,\s]*",
                            string.Empty);
                        var numbers = formattedValue.Trim().Replace(" ", ",").Split(',');
                        var sortedNumbers =
                            numbers.Distinct().Except(new[] {string.Empty}).Select(int.Parse).OrderBy(x => x).ToList();

                        if (CurrentItem != null && i == currentPlayIndex)
                        {
                            if (sortedNumbers.Any(num => num >= Media.Chapters.Count))
                            {
                                if (Media.Chapters.Count == 0) ShowCellTooltip(skipChapterCell, "This file has no chapters");
                                else
                                {
                                    ShowCellTooltip(skipChapterCell,
                                        "Only numbers < " + Media.Chapters.Count + " are allowed");
                                }

                                sortedNumbers.RemoveAll(num => num >= Media.Chapters.Count);
                            }
                            if (Media.Chapters.Count == 0) sortedNumbers.Clear();
                        }

                        formattedValue = String.Join(",", sortedNumbers);
                        skipChapterCell.Value = formattedValue;
                    }

                    if (endChapterCell.Value != null && endChapterCell.Value.ToString() != string.Empty)
                    {
                        var value = new String(endChapterCell.Value.ToString().Where(Char.IsDigit).ToArray());

                        if (CurrentItem != null && i == currentPlayIndex)
                        {
                            if (value.Length > 0 && int.Parse(value) > Media.Chapters.Count)
                            {
                                if (Media.Chapters.Count == 0) ShowCellTooltip(endChapterCell, "This file has no chapters");
                                else
                                {
                                    ShowCellTooltip(endChapterCell,
                                        "Only numbers <= " + Media.Chapters.Count + " are allowed");
                                }

                                value = Media.Chapters.Count.ToString();
                            }
                            if (Media.Chapters.Count == 0) value = string.Empty;
                        }

                        endChapterCell.Value = value;
                    }
                }

                UpdatePlaylist();
            }
            catch (Exception ex)
            {
                Player.HandleException(ex);
            }
        }

        private void ParseWithoutChapters(string line)
        {
            string title = string.Empty;
            var isActive = false;

            if (line.StartsWith(ActiveIndicator))
            {
                title = line.Substring(ActiveIndicator.Length).Trim();
                isActive = true;
            }
            else if (line.StartsWith(InactiveIndicator)) title = line.Substring(InactiveIndicator.Length).Trim();
            else throw new FileLoadException();

            var item = new PlaylistItem(title, isActive);
            Playlist.Add(item);

            Task.Factory.StartNew(GetMediaDuration);
        }

        private void ParseWithChapters(string line)
        {
            var splitLine = line.Split('|');
            string title = string.Empty;
            var isActive = false;
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
            else throw new FileLoadException();

            if (splitLine[1].Length > 0)
            {
                splitLine[1] = splitLine[1].Substring(splitLine[1].IndexOf(':') + 1).Trim();
                skipChapters = new List<int>(splitLine[1].Split(',').Select(int.Parse));
            }

            int endChapter = int.Parse(splitLine[2].Substring(splitLine[2].IndexOf(':') + 1).Trim());
            Playlist.Add(new PlaylistItem(title, skipChapters, endChapter, isActive));

            Task.Factory.StartNew(GetMediaDuration);
        }

        private void UpdatePlaylist()
        {
            try
            {
                for (var i = 0; i < dgv_PlayList.Rows.Count; i++)
                {
                    var skipChapters = new List<int>();
                    int endChapter = -1;

                    var skipChapterCell = dgv_PlayList.Rows[i].Cells[skipCellIndex];
                    var endChapterCell = dgv_PlayList.Rows[i].Cells[endCellIndex];

                    if (skipChapterCell.Value != null && skipChapterCell.Value.ToString() != string.Empty)
                    {
                        skipChapters = skipChapterCell.Value.ToString().Split(',').Select(int.Parse).ToList();
                        Playlist.ElementAt(i).HasChapter = true;
                    }

                    if (endChapterCell.Value != null && endChapterCell.Value.ToString() != string.Empty) endChapter = int.Parse(endChapterCell.Value.ToString());

                    Playlist.ElementAt(i).SkipChapters = skipChapters;
                    Playlist.ElementAt(i).EndChapter = endChapter;
                }
            }
            catch (Exception ex)
            {
                Player.HandleException(ex);
            }
        }

        #endregion

        #region Media Methods

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

        public void PlayNextFileInDirectory(bool next = true)
        {
            if (Player.State == PlayerState.Closed) return;

            string mediaPath = Media.FilePath;
            string mediaDir = Path.GetDirectoryName(mediaPath);
            var mediaFiles = playListUi.GetMediaFiles(mediaDir);
            string nextFile = next
                ? mediaFiles.SkipWhile(file => file != mediaPath).Skip(1).FirstOrDefault()
                : mediaFiles.TakeWhile(file => file != mediaPath).LastOrDefault();

            if (nextFile == null) return;

            Media.Open(nextFile);

            if (Playlist.Count == 1) ActiveFile(nextFile);
            else OpenFiles(new[] {nextFile});
        }

        private void OpenMedia(bool queue = false)
        {
            if (currentPlayIndex < 0 || currentPlayIndex >= Playlist.Count) return;

            bool playerWasFullScreen = Player.FullScreenMode.Active;
            ResetActive();

            try
            {
                var item = Playlist[currentPlayIndex];
                dgv_PlayList.CurrentCell = dgv_PlayList.Rows[currentPlayIndex].Cells[titleCellIndex];

                if (File.Exists(item.FilePath))
                {
                    SetPlayStyling();
                    Media.Open(item.FilePath);
                    if (queue) Media.Stop();
                }
                else
                {
                    if (currentPlayIndex != Playlist.Count - 1) PlayNext();
                    else CloseMedia();

                    SetPlayStyling();
                    return;
                }

                if (playerWasFullScreen) Player.FullScreenMode.Active = true;

                item.Active = true;
                CurrentItem = item;
                previousChapterPosition = 0;

                if (!queue) Text = Player.State + " ─ " + CurrentItem.FilePath;

                ParseChapterInput();
            }
            catch (Exception ex)
            {
                Player.HandleException(ex);
                PlayNext();
            }

            Task.Factory.StartNew(GetCurrentMediaDuration);
            dgv_PlayList.Invalidate();
        }

        public void CloseMedia()
        {
            CurrentItem = null;
            currentPlayIndex = -1;
            Text = "Playlist";
            Media.Close();
            dgv_PlayList.Invalidate();
        }

        #endregion

        #region The Methods

        public void InsertFile(int index, string fileName)
        {
            var item = new PlaylistItem(fileName, false);
            Playlist.Insert(index, item);
            PopulatePlaylist();
        }

        public void AddFiles(string[] fileNames)
        {
            AddFilesToPlaylist(fileNames);
            if (Player.State == PlayerState.Playing || Player.State == PlayerState.Paused) return;
            currentPlayIndex = fileNames.Count() > 1 ? Playlist.Count - fileNames.Count() : Playlist.Count - 1;
            OpenMedia(true);
        }

        public void ActiveFile(string fileName)
        {
            ResetActive();
            var item = new PlaylistItem(fileName, true);
            ClearPlaylist();
            Playlist.Add(item);
            CurrentItem = item;
            PopulatePlaylist();

            Text = Player.State + " ─ " + CurrentItem.FilePath;

            Task.Factory.StartNew(GetCurrentMediaDuration);
        }

        private void AddFilesToPlaylist(string[] fileNames)
        {
            foreach (var item in fileNames.Select(s => new PlaylistItem(s, false) {EndChapter = -1}))
            {
                Playlist.Add(item);
            }

            if (dgv_PlayList.CurrentRow != null) selectedRowIndex = dgv_PlayList.CurrentRow.Index;

            PopulatePlaylist();

            if (selectedRowIndex < 0) selectedRowIndex = 0;
            else if (selectedRowIndex > Playlist.Count - 1) selectedRowIndex = Playlist.Count - 1;

            dgv_PlayList.CurrentCell = dgv_PlayList.Rows[selectedRowIndex].Cells[titleCellIndex];

            Task.Factory.StartNew(GetMediaDuration);
        }

        private void AddFolderToPlaylist()
        {
            using (var fd = new VistaFolderBrowserDialog())
            {
                fd.Description = "Add folder to playlist";
                fd.UseDescriptionForTitle = true;
                fd.ShowNewFolderButton = true;

                if (fd.ShowDialog(this) != DialogResult.OK) return;

                var media = playListUi.GetAllMediaFiles(fd.SelectedPath);
                if (media.ToArray().Length == 0)
                {
                    MessageBox.Show("There are no files in the selected directory.", "Warning", MessageBoxButtons.OK,
                        MessageBoxIcon.Warning);
                    return;
                }

                AddFiles(media.ToArray());
            }
        }

        private void AddClipboardToPlaylist()
        {
            var files = Clipboard.GetText().Replace("\r", string.Empty).Split('\n').ToList();
            files.RemoveAll(f => !File.Exists(f));

            if (files.Count < 1) return;
            AddFiles(files.ToArray());
        }

        public void OpenFiles(string[] fileNames)
        {
            AddFilesToPlaylist(fileNames);
            currentPlayIndex = fileNames.Count() > 1 ? Playlist.Count - fileNames.Count() : Playlist.Count - 1;
            OpenMedia();
        }

        private void OpenFolder()
        {
            ClearPlaylist();

            using (var fd = new VistaFolderBrowserDialog())
            {
                fd.Description = "Open and play folder";
                fd.UseDescriptionForTitle = true;
                fd.ShowNewFolderButton = true;

                if (fd.ShowDialog(this) != DialogResult.OK) return;

                var media = playListUi.GetAllMediaFiles(fd.SelectedPath);
                if (media.ToArray().Length == 0)
                {
                    MessageBox.Show("There are no files in the selected directory.", "Warning", MessageBoxButtons.OK,
                        MessageBoxIcon.Warning);
                    return;
                }

                OpenFiles(media.ToArray());
            }
        }

        private void OpenClipboard()
        {
            ClearPlaylist();

            var files = Clipboard.GetText().Replace("\r", string.Empty).Split('\n').ToList();
            files.RemoveAll(f => !File.Exists(f));

            if (files.Count < 1) return;
            OpenFiles(files.ToArray());
        }

        public void RemoveFile(int index)
        {
            Playlist.RemoveAt(index);
            PopulatePlaylist();
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
                    if (index == currentPlayIndex) CloseMedia();

                    Playlist.RemoveAt(index);
                }

                PopulatePlaylist();

                selectedRowIndex = selectedRowIndex < 0
                    ? 0
                    : selectedRowIndex > Playlist.Count - 1 ? Playlist.Count - 1 : selectedRowIndex;

                dgv_PlayList.CurrentCell = Playlist.Count > 0
                    ? dgv_PlayList.Rows[selectedRowIndex].Cells[titleCellIndex]
                    : dgv_PlayList.CurrentCell = null;
            }
            catch (Exception ex)
            {
                Player.HandleException(ex);
            }
        }

        private void RemoveUnselectedItems()
        {
            var rowIndexes = new List<int>();

            try
            {
                if (Playlist.Count <= 0) return;
                if (dgv_PlayList.CurrentRow != null) selectedRowIndex = dgv_PlayList.CurrentRow.Index;

                rowIndexes.AddRange(
                    dgv_PlayList.Rows.Cast<DataGridViewRow>().Where(r1 => !r1.Selected).Select(r2 => r2.Index));

                foreach (int index in rowIndexes.OrderByDescending(v => v))
                {
                    if (index == currentPlayIndex) CloseMedia();

                    Playlist.RemoveAt(index);
                }

                PopulatePlaylist();

                selectedRowIndex = selectedRowIndex < 0
                    ? 0
                    : selectedRowIndex > Playlist.Count - 1 ? Playlist.Count - 1 : selectedRowIndex;

                dgv_PlayList.CurrentCell = Playlist.Count > 0
                    ? dgv_PlayList.Rows[selectedRowIndex].Cells[titleCellIndex]
                    : dgv_PlayList.CurrentCell = null;
            }
            catch (Exception ex)
            {
                Player.HandleException(ex);
            }
        }

        private void RemoveNonExistentItems()
        {
            try
            {
                if (Playlist.Count <= 0) return;
                if (dgv_PlayList.CurrentRow != null) selectedRowIndex = dgv_PlayList.CurrentRow.Index;

                Playlist.RemoveAll(p => !File.Exists(p.FilePath));

                PopulatePlaylist();

                selectedRowIndex = selectedRowIndex < 0
                    ? 0
                    : selectedRowIndex > Playlist.Count - 1 ? Playlist.Count - 1 : selectedRowIndex;

                dgv_PlayList.CurrentCell = Playlist.Count > 0
                    ? dgv_PlayList.Rows[selectedRowIndex].Cells[titleCellIndex]
                    : dgv_PlayList.CurrentCell = null;
            }
            catch (Exception ex)
            {
                Player.HandleException(ex);
            }
        }

        private void ViewFileLocation()
        {
            if (Playlist.Count == 0) return;
            Process.Start(Path.GetDirectoryName(Playlist[dgv_PlayList.CurrentRow.Index].FilePath));
        }

        private void ViewMediaInfo()
        {
            string media = Playlist[dgv_PlayList.CurrentRow.Index].FilePath;
            var mediaInfo = new ViewMediaInfoForm(media);
            mediaInfo.ShowDialog();
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
                    .ThenByDescending(f => Path.GetFileName(f.FilePath), new NaturalSortComparer()).ToList();
            }

            PopulatePlaylist();
        }

        public void RememberPlaylist()
        {
            TempRememberedFiles.Clear();
            if (Playlist.Count <= 0) return;

            foreach (var i in Playlist)
            {
                string skipChapters = string.Empty;

                if (i.SkipChapters != null && i.SkipChapters.Count > 0) skipChapters = string.Join(",", i.SkipChapters);

                TempRememberedFiles.Add(i.FilePath + "|" + skipChapters + "|" + i.EndChapter + "|" +
                                        i.Active);
            }
        }

        public void RestoreRememberedPlaylist()
        {
            var playList = new List<PlaylistItem>();

            foreach (string f in TempRememberedFiles)
            {
                var s = f.Split('|');
                string filePath = s[0];
                var skipChapters = new List<int>();
                if (s[1].Length > 0)
                {
                    if (s[1].Contains(",")) skipChapters = s[1].Split(',').Select(int.Parse).ToList();
                    else skipChapters.Add(int.Parse(s[1]));
                }
                int endChapter = int.Parse(s[2]);
                bool active = Boolean.Parse(s[3]);

                playList.Add(new PlaylistItem(filePath, skipChapters, endChapter, active));
            }

            Playlist = playList;
            PopulatePlaylist();
            RefreshPlaylist();
        }

        private void ShufflePlayList()
        {
            RememberPlaylist();
            Playlist.Shuffle();
            PopulatePlaylist();
        }

        #endregion

        #region Controls Handling Methods

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
                savePlaylistAsToolStripMenuItem.Enabled = true;
                viewFileLocationToolStripMenuItem.Enabled = true;
                viewMediaInfoToolStripMenuItem.Enabled = true;
                playToolStripMenuItem.Enabled = true;
                removeSelectedItemsToolStripMenuItem.Enabled = true;
                removeUnselectedItemsToolStripMenuItem.Enabled = true;
                removeNonExistentItemsToolStripMenuItem.Enabled = true;
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
                savePlaylistAsToolStripMenuItem.Enabled = false;
                viewFileLocationToolStripMenuItem.Enabled = false;
                viewMediaInfoToolStripMenuItem.Enabled = false;
                playToolStripMenuItem.Enabled = false;
                removeSelectedItemsToolStripMenuItem.Enabled = false;
                removeUnselectedItemsToolStripMenuItem.Enabled = false;
                removeNonExistentItemsToolStripMenuItem.Enabled = false;
                buttonNew.BackgroundImage = buttonNewDisabled.BackgroundImage;
                buttonSave.BackgroundImage = buttonSaveDisabled.BackgroundImage;
                buttonDel.BackgroundImage = buttonDelDisabled.BackgroundImage;
            }
        }

        private void HandleContextMenu()
        {
            if (dgv_PlayList.Rows.Count < 1) return;

            if (dgv_PlayList.CurrentCell.RowIndex != currentPlayIndex) playToolStripMenuItem.Text = "Play";
            else
            {
                switch (Player.State)
                {
                    case PlayerState.Paused:
                        playToolStripMenuItem.Text = "Resume";
                        break;
                    case PlayerState.Playing:
                        playToolStripMenuItem.Text = "Pause";
                        break;
                    default:
                        playToolStripMenuItem.Text = "Play";
                        break;
                }
            }
        }

        #endregion

        #region Column Handling Methods

        private void SetColumnSize()
        {
            if (columnsFixed) return;
            if (Columns == null || Columns.Count == 0) return;

            for (var i = 0; i < dgv_PlayList.Columns.Count; i++)
            {
                var c = dgv_PlayList.Columns[i];
                var split = Columns[i].Split('|');
                if (split[0] != c.Name) continue;
                if (split[0] != "Title") c.Visible = Convert.ToBoolean(split[1]);

                c.Width = int.Parse(split[2]);
                c.FillWeight = int.Parse(split[2]);
            }

            columnsFixed = true;
        }

        private void SetPlaylistToFill()
        {
            foreach (
                var c in
                    from DataGridViewColumn c in dgv_PlayList.Columns where c.Name != "Playing" select c)
            {
                c.AutoSizeMode = DataGridViewAutoSizeColumnMode.Fill;
            }
        }

        private void FitColumnsToHeader()
        {
            var list = new int[7];

            for (var i = 1; i < dgv_PlayList.Columns.Count; i++)
            {
                var c = dgv_PlayList.Columns[i];
                c.AutoSizeMode = DataGridViewAutoSizeColumnMode.ColumnHeader;
                list[i - 1] = c.Width;
            }

            for (var i = 1; i < dgv_PlayList.Columns.Count; i++)
            {
                var c = dgv_PlayList.Columns[i];
                c.AutoSizeMode = DataGridViewAutoSizeColumnMode.Fill;
                c.MinimumWidth = list[i - 1];
            }
        }

        private void SetColumnStates()
        {
            numberToolStripMenuItem.Checked = Number.Visible;
            directoryToolStripMenuItem.Checked = CurrentDirectory.Visible;
            fullPathToolStripMenuItem.Checked = FullPath.Visible;
            skipChaptersToolStripMenuItem.Checked = SkipChapters.Visible;
            endChapterToolStripMenuItem.Checked = EndChapter.Visible;
            durationToolStripMenuItem.Checked = Duration.Visible;

            if (Title.Visible) titleCellIndex = Title.Index;
            if (SkipChapters.Visible) skipCellIndex = SkipChapters.Index;
            if (EndChapter.Visible) endCellIndex = EndChapter.Index;
        }

        private void UpdateColumns(object sender, EventArgs e)
        {
            Number.Visible = numberToolStripMenuItem.Checked;
            FullPath.Visible = fullPathToolStripMenuItem.Checked;
            CurrentDirectory.Visible = directoryToolStripMenuItem.Checked;
            SkipChapters.Visible = skipChaptersToolStripMenuItem.Checked;
            EndChapter.Visible = endChapterToolStripMenuItem.Checked;
            Duration.Visible = durationToolStripMenuItem.Checked;

            if (Title.Visible) titleCellIndex = Title.Index;
            if (SkipChapters.Visible) skipCellIndex = SkipChapters.Index;
            if (EndChapter.Visible) endCellIndex = EndChapter.Index;
        }

        #endregion

        #region Helper Methods

        private void ResetActive()
        {
            foreach (var item in Playlist)
            {
                item.Active = false;
            }
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

            if (currentPlayIndex == -1) return;
            dgv_PlayList.Rows[currentPlayIndex].DefaultCellStyle.ForeColor = Color.White;
            dgv_PlayList.Rows[currentPlayIndex].Selected = true;
        }

        private static void SelectChapter(int chapterNum)
        {
            if (Player.State == PlayerState.Closed) return;

            var chapters = GetChapters();

            if (chapters.ElementAt(chapterNum) == null) return;
            Media.Seek(chapters.ElementAt(chapterNum).Position);
            Player.OsdText.Show(chapters.ElementAt(chapterNum).Name);
        }

        private static int GetChapterIndexByPosition(long position)
        {
            var currentChapterIndex = 0;

            foreach (var c in GetChapters().Where(c => c != null))
            {
                currentChapterIndex++;
                if (c.Position != position) continue;
                return currentChapterIndex;
            }

            return 0;
        }

        private static IEnumerable<Chapter> GetChapters()
        {
            return Media.Chapters.OrderBy(chapter => chapter.Position);
        }

        private static void DisableTabStop(Control c)
        {
            if (c.GetType() == typeof(DataGridView)) return;
            c.TabStop = false;

            foreach (Control i in c.Controls)
            {
                DisableTabStop(i);
            }
        }

        private void SelectNextEditableCell()
        {
            var currentCell = dgv_PlayList.CurrentCell;
            if (currentCell == null) return;

            int nextRow = currentCell.RowIndex;
            var nextCell = SkipChapters.Visible
                ? dgv_PlayList.Rows[nextRow].Cells[skipCellIndex]
                : dgv_PlayList.Rows[nextRow].Cells[endCellIndex];

            if (nextCell == null || !nextCell.Visible) return;
            dgv_PlayList.CurrentCell = nextCell;
        }

        private void ShowCellTooltip(DataGridViewCell cell, string message)
        {
            var toolTip = new ToolTip();
            var cellDisplayRect = dgv_PlayList.GetCellDisplayRectangle(cell.ColumnIndex, cell.RowIndex, false);
            toolTip.Show(message, dgv_PlayList,
                cellDisplayRect.X + cell.Size.Width / 2,
                cellDisplayRect.Y + cell.Size.Height / 2,
                2000);
        }

        private void ShowCurrentCellTooltip(string message)
        {
            var toolTip = new ToolTip();
            var cell = dgv_PlayList.CurrentCell;
            var cellDisplayRect = dgv_PlayList.GetCellDisplayRectangle(cell.ColumnIndex, cell.RowIndex, false);
            toolTip.Show(message, dgv_PlayList,
                cellDisplayRect.X + cell.Size.Width / 2,
                cellDisplayRect.Y + cell.Size.Height / 2,
                2000);
        }

        #endregion

        #region PlayerControl Events

        private void PlaylistFormClosing(object sender, FormClosingEventArgs e)
        {
            if (e.CloseReason != CloseReason.UserClosing) return;
            e.Cancel = true;
            Hide();
            timer.Enabled = false;
        }

        private void PlayerStateChanged(object sender, PlayerStateEventArgs e)
        {
            if (String.IsNullOrEmpty(Media.FilePath)) return;
            if (!File.Exists(Media.FilePath))
            {
                currentPlayIndex = -1;
                Text = "Playlist";
                RefreshPlaylist();
                return;
            }

            if (CurrentItem == null) return;
            Text = Player.State + " - " + CurrentItem.FilePath;

            HandleContextMenu();

            if (currentPlayIndex == -1) return;
            dgv_PlayList.InvalidateRow(currentPlayIndex);
        }

        private void PlaybackCompleted(object sender, EventArgs e)
        {
            if (Player.State == PlayerState.Closed) return;
            if (Media.Position == Media.Duration) PlayNext();
        }

        private void FrameDecoded(object sender, FrameEventArgs e)
        {
            if (Media.FilePath != string.Empty && Media.Chapters.Count != 0 && CurrentItem != null &&
                CurrentItem.HasChapter)
            {
                previousChapterPosition =
                    GetChapters()
                        .Aggregate(
                            (prev, next) => e.SampleTime >= prev.Position && e.SampleTime <= next.Position ? prev : next)
                        .Position;
            }
        }

        private void FramePresented(object sender, FrameEventArgs e)
        {
            if (Media.FilePath != string.Empty && Media.Chapters.Count != 0 && CurrentItem != null &&
                CurrentItem.HasChapter)
            {
                if (e.SampleTime >= previousChapterPosition)
                {
                    int currentChapterIndex = GetChapterIndexByPosition(previousChapterPosition);

                    if (CurrentItem.SkipChapters.Contains(currentChapterIndex) &&
                        currentChapterIndex < Media.Chapters.Count) SelectChapter(currentChapterIndex);
                    if (currentChapterIndex == CurrentItem.EndChapter) PlayNext();
                }
            }
        }

        private void EnteringFullScreenMode(object sender, EventArgs e)
        {
            wasShowing = Visible;
            Hide();
        }

        private void ExitedFullScreenMode(object sender, EventArgs e)
        {
            if (wasShowing) Show(Gui.VideoBox);
        }

        #endregion

        #region Playlist Datagridview Events

        private void dgv_PlayList_CellFormatting(object sender, DataGridViewCellFormattingEventArgs e)
        {
            var skipChapterCell = dgv_PlayList.Rows[e.RowIndex].Cells[skipCellIndex];
            var endChapterCell = dgv_PlayList.Rows[e.RowIndex].Cells[endCellIndex];

            if (skipChapterCell.IsInEditMode || endChapterCell.IsInEditMode) e.CellStyle.ForeColor = Color.Black;
        }

        private void dgv_PlayList_CellPainting(object sender, DataGridViewCellPaintingEventArgs e)
        {
            e.Paint(e.ClipBounds, DataGridViewPaintParts.All);

            bool paintPlayRow = CurrentItem != null && e.RowIndex > -1 && e.RowIndex == currentPlayIndex;
            if (!paintPlayRow) return;

            var brush = new SolidBrush(Color.FromArgb(42, 127, 183));
            var icon = new Bitmap(24, 24);

            switch (Player.State)
            {
                case PlayerState.Playing:
                    icon = (Bitmap)PlayButton.BackgroundImage;
                    break;
                case PlayerState.Paused:
                    icon = (Bitmap)PauseButton.BackgroundImage;
                    break;
                case PlayerState.Stopped:
                    icon = (Bitmap)StopButton.BackgroundImage;
                    break;
                default:
                    brush = new SolidBrush(Color.FromArgb(0, 0, 0, 0));
                    icon = new Bitmap(24, 24);
                    break;
            }

            if (e.ColumnIndex == 0)
            {
                var rect = new Rectangle(e.CellBounds.X + 15, e.CellBounds.Y + 4, e.CellBounds.Width,
                    e.CellBounds.Height - 9);
                var offset = new Point(e.CellBounds.X, e.CellBounds.Y + 2);
                e.Graphics.FillRectangle(brush, rect);
                e.Graphics.DrawImage(icon, new Rectangle(offset, new Size(24, 24)), 0, 0, 24, 24, GraphicsUnit.Pixel);
            }
            else
            {
                var rect = new Rectangle(e.CellBounds.X, e.CellBounds.Y + 4, e.CellBounds.Width,
                    e.CellBounds.Height - 9);
                e.Graphics.FillRectangle(brush, rect);
            }

            e.Paint(e.ClipBounds, DataGridViewPaintParts.ContentForeground);
            e.Handled = true;
        }

        private void dgv_PlayList_RowsAdded(object sender, DataGridViewRowsAddedEventArgs e)
        {
            SetControlStates();
        }

        private void dgv_PlayList_RowsRemoved(object sender, DataGridViewRowsRemovedEventArgs e)
        {
            SetControlStates();
        }

        private void dgv_PlayList_CellDoubleClick(object sender, DataGridViewCellEventArgs e)
        {
            PlaySelectedFile();
        }

        private void dgv_PlayList_CellEndEdit(object sender, DataGridViewCellEventArgs e)
        {
            ParseChapterInput();
        }

        private void dgv_PlayList_SelectionChanged(object sender, EventArgs e)
        {
            if (Playlist.Count == 0) return;

            if (dgv_PlayList.SelectedRows.Count > 0)
            {
                openFileDialog.InitialDirectory =
                    Path.GetDirectoryName(Playlist[dgv_PlayList.SelectedRows[0].Index].FilePath);
            }
        }

        private void dgv_PlayList_EditingControlShowing(object sender, DataGridViewEditingControlShowingEventArgs e)
        {
            e.Control.KeyPress -= dgv_PlayList_HandleInput;
            if (dgv_PlayList.CurrentCell.ColumnIndex <= 1) return;

            var tb = e.Control as TextBox;
            if (tb != null) tb.KeyPress += dgv_PlayList_HandleInput;
        }

        private void dgv_PlayList_HandleInput(object sender, KeyPressEventArgs e)
        {
            if (!char.IsControl(e.KeyChar) && !char.IsDigit(e.KeyChar) && e.KeyChar != ',' && e.KeyChar != ' ' &&
                dgv_PlayList.CurrentCell.ColumnIndex == skipCellIndex)
            {
                ShowCurrentCellTooltip("Only numbers are allowed. You may separate them with a comma or a space.");
                e.Handled = true;
            }

            if (!char.IsControl(e.KeyChar) && !char.IsDigit(e.KeyChar) &&
                dgv_PlayList.CurrentCell.ColumnIndex == endCellIndex)
            {
                ShowCurrentCellTooltip("Only numbers are allowed.");
                e.Handled = true;
            }
        }

        private void dgv_Playlist_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyCode == Keys.Tab)
            {
                SelectNextEditableCell();
                e.SuppressKeyPress = true;
                e.Handled = true;
            }

            if (e.KeyCode == Keys.Delete) RemoveSelectedItems();

            if (e.KeyCode == Keys.Enter)
            {
                PlaySelectedFile();
                e.Handled = true;
            }
        }

        private void dgv_PlayList_MouseMove(object sender, MouseEventArgs e)
        {
            if (Playlist.Count < 2) return;
            if (e.Button != MouseButtons.Left) return;
            if (dragRowRect != Rectangle.Empty && !dragRowRect.Contains(e.X, e.Y) && isDragging) dgv_PlayList.DoDragDrop(dgv_PlayList.Rows[dragRowIndex], DragDropEffects.Move);
        }

        private void dgv_PlayList_MouseDown(object sender, MouseEventArgs e)
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
                    HandleContextMenu();
                    dgv_PlaylistContextMenu.Show(Cursor.Position);
                }
            }
        }

        private void dgv_PlayList_MouseUp(object sender, MouseEventArgs e)
        {
            dragRowIndex = dgv_PlayList.HitTest(e.X, e.Y).RowIndex;

            if (dragRowIndex != -1) isDragging = false;
        }

        private void dgv_PlayList_DragOver(object sender, DragEventArgs e)
        {
            e.Effect = DragDropEffects.Move;
        }

        private void dgv_PlayList_DragDrop(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                var files = (string[])e.Data.GetData(DataFormats.FileDrop);
                if (files.Length == 1)
                {
                    string filename = files[0];

                    if (Directory.Exists(filename))
                    {
                        var media = playListUi.GetAllMediaFiles(filename);
                        AddFiles(media.ToArray());
                        return;
                    }
                    if (PlayerExtensions.Playlist.Playlist.IsPlaylistFile(filename))
                    {
                        OpenPlaylist(filename);
                        return;
                    }
                }

                var mediaFiles = new List<string>();

                foreach (string p in files)
                {
                    var attr = File.GetAttributes(p);
                    bool isFolder = (attr & FileAttributes.Directory) == FileAttributes.Directory;

                    if (!isFolder) continue;
                    if (Directory.Exists(p)) mediaFiles.AddRange(playListUi.GetAllMediaFiles(p));
                }

                if (mediaFiles.Count > 0) AddFiles(mediaFiles.NaturalSort().ToArray());

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

        #endregion

        #region Button Events

        private void ButtonPlayClick(object sender, EventArgs e)
        {
            if (dgv_PlayList.CurrentCell.RowIndex != currentPlayIndex) PlaySelectedFile();
            else
            {
                switch (Player.State)
                {
                    case PlayerState.Paused:
                        Media.Play();
                        break;
                    case PlayerState.Playing:
                        Media.Pause();
                        break;
                    default:
                        PlaySelectedFile();
                        break;
                }
            }
        }

        private void ButtonAddFilesClick(object sender, EventArgs e)
        {
            if (openFileDialog.ShowDialog(this) != DialogResult.OK) return;

            var fileNames = openFileDialog.FileNames;

            AddFiles(fileNames);
            dgv_PlayList.Focus();
        }

        private void ButtonAddFolderClick(object sender, EventArgs e)
        {
            AddFolderToPlaylist();
            dgv_PlayList.Focus();
        }

        private void ButtonAddFromClipboardClick(object sender, EventArgs e)
        {
            AddClipboardToPlaylist();
            dgv_PlayList.Focus();
        }

        private void ButtonOpenFilesClick(object sender, EventArgs e)
        {
            ClearPlaylist();

            if (openFileDialog.ShowDialog(this) != DialogResult.OK) return;

            var fileNames = openFileDialog.FileNames;

            OpenFiles(fileNames);
            dgv_PlayList.Focus();
        }

        private void ButtonOpenFolderClick(object sender, EventArgs e)
        {
            OpenFolder();
            dgv_PlayList.Focus();
        }

        private void ButtonOpenFromClipboardClick(object sender, EventArgs e)
        {
            OpenClipboard();
            dgv_PlayList.Focus();
        }

        private void ButtonRemoveSelectedItemsClick(object sender, EventArgs e)
        {
            RemoveSelectedItems();
            dgv_PlayList.Focus();
        }

        private void ButtonRemoveUnselectedItemsClick(object sender, EventArgs e)
        {
            RemoveUnselectedItems();
            dgv_PlayList.Focus();
        }

        private void ButtonRemoveNonExistentItemsClick(object sender, EventArgs e)
        {
            RemoveNonExistentItems();
            dgv_PlayList.Focus();
        }

        private void ButtonNewPlaylistClick(object sender, EventArgs e)
        {
            NewPlaylist(true);
        }

        private void ButtonAddPlaylistClick(object sender, EventArgs e)
        {
            OpenPlaylist(false);
        }

        private void ButtonOpenPlaylistClick(object sender, EventArgs e)
        {
            OpenPlaylist();
        }

        private void ButtonSavePlaylistClick(object sender, EventArgs e)
        {
            SavePlaylist();
        }

        private void ButtonSavePlaylistAsClick(object sender, EventArgs e)
        {
            SavePlaylistAs();
        }

        private void ButtonLeftClick(object sender, EventArgs e)
        {
            PlayPrevious();
        }

        private void ButtonRightClick(object sender, EventArgs e)
        {
            PlayNext();
        }

        private void ButtonViewFileLocation(object sender, EventArgs e)
        {
            ViewFileLocation();
        }

        private void ButtonViewMediaInfo(object sender, EventArgs e)
        {
            ViewMediaInfo();
        }

        private void ButtonSortAscendingClick(object sender, EventArgs e)
        {
            SortPlayList();
        }

        private void ButtonSortDescendingClick(object sender, EventArgs e)
        {
            SortPlayList(false);
        }

        private void ButtonShuffleClick(object sender, EventArgs e)
        {
            ShufflePlayList();
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

        #endregion

        #region Form Events

        private void PlaylistForm_Load(object sender, EventArgs e)
        {
            SetColumnSize();
        }

        private void PlaylistForm_Shown(object sender, EventArgs e)
        {
            SetColumnSize();
            FitColumnsToHeader();
        }

        private void PlaylistForm_Resize(object sender, EventArgs e)
        {
            SetPlaylistToFill();
        }

        private void PlaylistForm_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyData == Keys.Escape) Hide();

            if (e.KeyCode == Keys.V && e.Modifiers == (Keys.Control)) AddClipboardToPlaylist();

            if (e.KeyCode == Keys.P && e.Modifiers == (Keys.Control | Keys.Alt))
            {
                playListUi.ViewPlaylist();
                e.SuppressKeyPress = true;
                e.Handled = true;
            }

            if (e.KeyCode == Keys.Tab && e.Modifiers == Keys.Control)
            {
                var form = Player.ActiveForm;
                if (Player.FullScreenMode.Active || form.ContainsFocus) return;
                form.Activate();
                Cursor.Position = new Point(form.Location.X + 100, form.Location.Y + 100);
                e.SuppressKeyPress = true;
                e.Handled = true;
            }
        }

        #endregion

        #region Threaded Methods

        public void GetCurrentMediaDuration()
        {
            try
            {
                var time = TimeSpan.FromMilliseconds(Media.Duration / 1000);
                CurrentItem.Duration = time.ToString(@"hh\:mm\:ss");
                Invoke((Action)(() =>
                {
                    if (dgv_PlayList.Rows.Count < 1) return;
                    dgv_PlayList.Rows[currentPlayIndex].Cells["Duration"].Value = time.ToString(@"hh\:mm\:ss");
                    dgv_PlayList.InvalidateRow(currentPlayIndex);
                }));
            }
            catch (Exception ex)
            {
                Player.HandleException(ex);
            }
        }

        public void GetMediaDuration()
        {
            try
            {
                for (var i = 0; i < Playlist.Count; i++)
                {
                    var item = Playlist[i];
                    if (!String.IsNullOrEmpty(item.Duration)) continue;
                    var media = new MediaFile(item.FilePath);
                    var time = TimeSpan.FromMilliseconds(media.duration);
                    item.Duration = time.ToString(@"hh\:mm\:ss");
                    Invoke((Action)(() =>
                    {
                        if (dgv_PlayList.Rows.Count < 1) return;
                        if (i != currentPlayIndex)
                        {
                            dgv_PlayList.Rows[i].Cells["Duration"].Value = time.ToString(@"hh\:mm\:ss");
                            dgv_PlayList.InvalidateRow(i);
                        }
                    }));
                }
            }
            catch (Exception ex)
            {
                Player.HandleException(ex);
            }
        }

        #endregion

        #region Timer Stuff

        private void HandleOpacity()
        {
            var pos = MousePosition;
            bool inForm = pos.X >= Left && pos.Y >= Top && pos.X < Right && pos.Y < Bottom;

            if (inForm || ActiveForm == this)
            {
                if (Opacity < MaxOpacity) Opacity += 0.1;
            }
            else if (Opacity > MinOpacity) Opacity -= 0.1;
        }

        private void TimerTick(object sender, EventArgs e)
        {
            HandleOpacity();
        }

        #endregion
    }

    #region PlaylistItem

    public class PlaylistItem
    {
        public string FilePath { get; set; }
        public bool Active { get; set; }
        public bool HasChapter { get; set; }
        public List<int> SkipChapters { get; set; }
        public int EndChapter { get; set; }
        public string Duration { get; set; }

        public PlaylistItem(string filePath, bool isActive)
        {
            if (string.IsNullOrWhiteSpace(filePath)) throw new ArgumentNullException("filePath");

            FilePath = filePath;
            Active = isActive;
        }

        public PlaylistItem(string filePath, List<int> skipChapter, int endChapter, bool isActive)
        {
            if (string.IsNullOrWhiteSpace(filePath)) throw new ArgumentNullException("filePath");

            FilePath = filePath;
            Active = isActive;
            SkipChapters = skipChapter;
            EndChapter = endChapter;
            HasChapter = true;
        }

        public PlaylistItem(string filePath, List<int> skipChapter, int endChapter, bool isActive, string duration)
        {
            if (string.IsNullOrWhiteSpace(filePath)) throw new ArgumentNullException("filePath");

            FilePath = filePath;
            Active = isActive;
            SkipChapters = skipChapter;
            EndChapter = endChapter;
            HasChapter = true;
            Duration = duration;
        }

        public override string ToString()
        {
            if (HasChapter)
            {
                return Path.GetFileName(FilePath) + " | SkipChapter: " + String.Join(",", SkipChapters) +
                       " | EndChapter: " + EndChapter ?? "???";
            }

            return Path.GetFileName(FilePath) ?? "???";
        }
    }

    #endregion

    #region ToolstripItem Proxy

    [ToolStripItemDesignerAvailability(ToolStripItemDesignerAvailability.StatusStrip)]
    public class ButtonStripItem : ToolStripControlHostProxy
    {
        public ButtonStripItem()
            : base(CreateButtonInstance()) {}

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
            : base(new Control()) {}

        public ToolStripControlHostProxy(Control c)
            : base(c) {}
    }

    #endregion

    #region Form base

    public class FormEx : Form
    {
        private float m_ScaleFactorHeight = -1f;
        private float m_ScaleFactorWidth = -1f;
        protected SizeF ScaleFactor { get; private set; }

        protected override void ScaleControl(SizeF factor, BoundsSpecified specified)
        {
            base.ScaleControl(factor, specified);

            if (!(m_ScaleFactorWidth < 0 || m_ScaleFactorHeight < 0)) return;

            if (m_ScaleFactorWidth < 0 && specified.HasFlag(BoundsSpecified.Width)) m_ScaleFactorWidth = factor.Width;
            if (m_ScaleFactorHeight < 0 && specified.HasFlag(BoundsSpecified.Height)) m_ScaleFactorHeight = factor.Height;

            if (m_ScaleFactorWidth < 0 || m_ScaleFactorHeight < 0) return;

            ScaleFactor = new SizeF(m_ScaleFactorWidth, m_ScaleFactorHeight);
        }
    }

    #endregion

    #region Natural Sorting

    [SuppressUnmanagedCodeSecurity]
    internal static class SafeNativeMethods
    {
        [DllImport("shlwapi.dll", CharSet = CharSet.Unicode)]
        public static extern int StrCmpLogicalW(string psz1, string psz2);
    }

    public class NaturalSortComparer : IComparer<string>
    {
        public NaturalSortComparer() : this(false) {}
        public NaturalSortComparer(bool descending) {}

        public int Compare(string a, string b)
        {
            var arrayA = a.Split(Path.DirectorySeparatorChar);
            var arrayB = b.Split(Path.DirectorySeparatorChar);

            int length = Math.Max(arrayA.Length, arrayB.Length);

            for (var i = 0; i < length; i++)
            {
                int result = SafeNativeMethods.StrCmpLogicalW(arrayA.Length > i ? arrayA[i].ToLower() : string.Empty,
                    arrayB.Length > i ? arrayB[i].ToLower() : string.Empty);

                if (result != 0) return result;
            }

            return 0;
        }
    }

    public static class ThreadSafeRandom
    {
        [ThreadStatic] private static Random Local;

        public static Random ThisThreadsRandom
        {
            get
            {
                return Local ??
                       (Local = new Random(unchecked(Environment.TickCount * 31 + Thread.CurrentThread.ManagedThreadId)));
            }
        }
    }

    #endregion

    #region Extensions Methods

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
                var value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }
    }

    #endregion
}
