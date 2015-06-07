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
using System.Windows.Forms;
using Mpdn.Extensions.Framework;

namespace Mpdn.Extensions.PlayerExtensions.Playlist
{
    public class Playlist : PlayerExtension<PlaylistSettings, PlaylistConfigDialog>
    {
        private const string Subcategory = "Playlist";

        private readonly PlaylistForm form = new PlaylistForm();

        private bool docked;
        
        private Form mpdnForm;
        private Point mpdnStartLocation;

        private Point formStartLocation;
        private Size formStartSize;
        private bool moving;
        private bool resizing;

        public PlaylistForm GetPlaylistForm
        {
            get { return form; }
        }

        public override ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("A1997E34-D67B-43BB-8FE6-55A71AE7184B"),
                    Name = "Playlist",
                    Description = "Adds playlist support with advanced capabilities",
                    Copyright = "Enhanced by Garteal"
                };
            }
        }

        public override void Initialize()
        {
            base.Initialize();
            form.Setup(this);

            PlayerControl.PlayerStateChanged += OnPlayerStateChanged;
            PlayerControl.PlaybackCompleted += OnPlaybackCompleted;
            PlayerControl.FormClosed += OnMpdnFormClosed;
            PlayerControl.DragEnter += OnDragEnter;
            PlayerControl.DragDrop += OnDragDrop;
            PlayerControl.CommandLineFileOpen += OnCommandLineFileOpen;
            mpdnForm = PlayerControl.Form;
            mpdnForm.Move += OnMpdnFormMove;
            mpdnForm.KeyDown += OnMpdnFormKeyDown;
            mpdnForm.MainMenuStrip.MenuActivate += OnMpdnFormMainMenuActivated;
            mpdnForm.SizeChanged += OnMpdnFormSizeChanged;
            mpdnForm.ResizeBegin += OnMpdnFormResizeBegin;
            mpdnForm.ResizeEnd += OnMpdnFormResizeEnd;
            form.VisibleChanged += OnFormVisibilityChanged;
            form.Move += OnFormMove;
            form.SizeChanged += OnFormSizeChanged;

            if (Settings.RememberWindowPosition)
            {
                form.RememberWindowPosition = Settings.RememberWindowPosition;
                form.WindowPosition = Settings.WindowPosition;
            }

            if (Settings.RememberWindowSize)
            {
                form.RememberWindowSize = Settings.RememberWindowSize;
                form.WindowSize = Settings.WindowSize;
                formStartSize = form.Size;
            }

            if (Settings.LockWindowSize)
            {
                form.LockWindowSize = Settings.LockWindowSize;
                SetFormToFixed();
            }

            if (Settings.SnapAndScaleWithPlayer)
            {
                form.SnapAndScaleWithPlayer = Settings.SnapAndScaleWithPlayer;
                SnapPlayer();
            }

            if (Settings.RememberColumns)
            {
                if (Settings.Columns != null && Settings.Columns.Count > 0)
                {
                    form.Columns = Settings.Columns;
                }
            }

            if (Settings.ShowPlaylistOnStartup)
            {
                ViewPlaylist();
            }

            if (Settings.BeginPlaybackOnStartup)
            {
                form.BeginPlaybackOnStartup = Settings.BeginPlaybackOnStartup;
            }

            if (Settings.BeginPlaybackWhenFileIsAdded)
            {
                form.BeginPlaybackWhenFileIsAdded = Settings.BeginPlaybackWhenFileIsAdded;
            }

            if (Settings.BeginPlaybackWhenPlaylistFileIsOpened)
            {
                form.BeginPlaybackWhenPlaylistFileIsOpened = Settings.BeginPlaybackWhenPlaylistFileIsOpened;
            }

            if (Settings.RememberPlaylist)
            {
                if (Settings.RememberedFiles.Count > 0)
                {
                    List<PlaylistItem> playList = new List<PlaylistItem>();

                    foreach (var f in Settings.RememberedFiles)
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

                    int activeItem = (playList.FindIndex(i => i.Active) > -1) ? playList.FindIndex(i => i.Active) : 0;

                    form.Playlist = playList;
                    form.PopulatePlaylist();
                    form.RefreshPlaylist();
                    form.FocusPlaylistItem(activeItem);

                    if (Settings.BeginPlaybackOnStartup)
                    {
                        form.PlayActive();
                    }
                }
            }

            FixFormLocationBounds();
        }

        public void Reinitialize()
        {
            if (Settings.LockWindowSize) SetFormToFixed();
            else SetFormToSizable();
            if (Settings.SnapAndScaleWithPlayer) SnapPlayer();

            form.RememberWindowPosition = Settings.RememberWindowPosition;
            form.RememberWindowSize = Settings.RememberWindowSize;
            form.SnapAndScaleWithPlayer = Settings.SnapAndScaleWithPlayer;
            form.LockWindowSize = Settings.LockWindowSize;
            form.BeginPlaybackOnStartup = Settings.BeginPlaybackOnStartup;
            form.BeginPlaybackWhenPlaylistFileIsOpened = Settings.BeginPlaybackWhenPlaylistFileIsOpened;
            form.BeginPlaybackWhenFileIsAdded = Settings.BeginPlaybackWhenFileIsAdded;
        }

        public override void Destroy()
        {
            PlayerControl.PlayerStateChanged -= OnPlayerStateChanged;
            PlayerControl.PlaybackCompleted -= OnPlaybackCompleted;
            PlayerControl.FormClosed -= OnMpdnFormClosed;
            PlayerControl.DragEnter -= OnDragEnter;
            PlayerControl.DragDrop -= OnDragDrop;
            PlayerControl.CommandLineFileOpen -= OnCommandLineFileOpen;
            mpdnForm = PlayerControl.Form;
            mpdnForm.Move -= OnMpdnFormMove;
            mpdnForm.KeyDown -= OnMpdnFormKeyDown;
            mpdnForm.MainMenuStrip.MenuActivate -= OnMpdnFormMainMenuActivated;
            mpdnForm.SizeChanged -= OnMpdnFormSizeChanged;
            mpdnForm.ResizeBegin -= OnMpdnFormResizeBegin;
            mpdnForm.ResizeEnd -= OnMpdnFormResizeEnd;
            form.VisibleChanged -= OnFormVisibilityChanged;
            form.Move -= OnFormMove;
            form.SizeChanged -= OnFormSizeChanged;

            base.Destroy();
            form.Dispose();
        }

        private void OnMpdnFormMainMenuActivated(object sender, EventArgs e)
        {
            foreach (ToolStripMenuItem item in mpdnForm.MainMenuStrip.Items)
            {
                if (item.DropDownItems[0].Name == "mmenuQuickOpen")
                {
                    item.DropDownItems[0].Click -= OnMpdnFormOpenClick;
                    item.DropDownItems[0].Click += OnMpdnFormOpenClick;
                }

                if (item.DropDownItems[1].Name == "openToolStripMenuItem")
                {
                    item.DropDownItems[1].Click -= OnMpdnFormOpenClick;
                    item.DropDownItems[1].Click += OnMpdnFormOpenClick;
                }

                if (item.DropDownItems[2].Name == "mmenuClose")
                {
                    item.DropDownItems[2].Click -= OnFormCloseMedia;
                    item.DropDownItems[2].Click += OnFormCloseMedia;
                }
            }
        }

        private void OnMpdnFormKeyDown(object sender, KeyEventArgs e)
        {
            if (e.Modifiers == Keys.Control && e.KeyCode == Keys.C)
            {
                CloseMedia();
            }

            if (e.Modifiers == Keys.Control && e.KeyCode == Keys.Tab)
            {
                if (!PlayerControl.InFullScreenMode && form.Visible && !form.ContainsFocus)
                {
                    form.Activate();
                    form.FocusPlaylist();
                    Cursor.Position = new Point(form.Location.X + 100, form.Location.Y + 100);
                    e.SuppressKeyPress = true;
                }
            }
        }

        private void OnFormCloseMedia(object sender, EventArgs e)
        {
            CloseMedia();
        }

        private void OnMpdnFormOpenClick(object sender, EventArgs e)
        {
            NewPlaylist();
        }

        private void OnPlayerStateChanged(object sender, EventArgs e)
        {
            if (Settings.AddToPlaylistOnFileOpen)
            {
                SetActiveFile();
            }
        }

        private void OnPlaybackCompleted(object sender, EventArgs e)
        {
            if (Settings.AddToPlaylistOnFileOpen && form.Playlist.Count > 1)
            {
                AddFileToPlaylist();
            }
            if (Settings.AfterPlaybackOpt == AfterPlaybackSettingsOpt.PlayNextFileInFolder)
            {
                PlayNextInFolder();
            }
            if (Settings.AfterPlaybackOpt == AfterPlaybackSettingsOpt.ClosePlayer)
            {
                CloseMPDN();
            }
        }

        private void OnFormVisibilityChanged(object sender, EventArgs e)
        {
            mpdnForm.BringToFront();
        }

        private void OnMpdnFormClosed(object sender, EventArgs e)
        {
            Settings.WindowPosition = form.Location;
            Settings.WindowSize = form.Size;

            Settings.Columns.Clear();

            for (int i = 0; i < form.GetDgvPlaylist().Columns.Count; i++)
            {
                var c = form.GetDgvPlaylist().Columns[i];
                Settings.Columns.Add(c.Name + "|"
                    + c.Visible + "|" + c.Width);
            }

            if (Settings.RememberPlaylist)
            {
                Settings.RememberedFiles.Clear();

                if (form.Playlist.Count > 0)
                {
                    foreach (PlaylistItem i in form.Playlist)
                    {
                        string skipChapters = "";

                        if (i.SkipChapters != null && i.SkipChapters.Count > 0)
                        {
                            skipChapters = string.Join(",", i.SkipChapters);
                        }

                        Settings.RememberedFiles.Add(i.FilePath + "|" + skipChapters + "|" + i.EndChapter + "|" +
                                                     i.Active);
                    }
                }
            }
        }

        private void OnFormMove(object sender, EventArgs e)
        {
            form.Refresh();

            if (moving)
                return;

            if (mpdnForm.WindowState != FormWindowState.Minimized || form.WindowState != FormWindowState.Minimized)
            {
                mpdnStartLocation = PlayerControl.Form.Location;
                formStartLocation = form.Location;
            }
        }

        private void OnMpdnFormMove(object sender, EventArgs e)
        {
            if (form.WindowState != FormWindowState.Minimized)
            {
                if (CursorIsOnResizeAnchor()) return;
                moving = true;
                form.Left = formStartLocation.X + PlayerControl.Form.Location.X - mpdnStartLocation.X;
                form.Top = formStartLocation.Y + PlayerControl.Form.Location.Y - mpdnStartLocation.Y;
                moving = false;
            }
        }

        void OnMpdnFormSizeChanged(object sender, EventArgs e)
        {
            Screen scn = Screen.FromPoint(mpdnForm.Location);

            if (Settings.SnapAndScaleWithPlayer) SnapPlayer();

            if (mpdnForm.WindowState == FormWindowState.Minimized)
            {
                form.Bounds = form.RestoreBounds;
            }

            if (mpdnForm.Left == 0 && mpdnForm.Height == scn.WorkingArea.Height)
            {
                int borderWidth = SystemInformation.SizingBorderWidth;

                docked = true;
                form.Width = scn.WorkingArea.Width / 2;
                form.Height = scn.WorkingArea.Height;
                if (Settings.LockWindowSize) form.Left = scn.WorkingArea.Right - form.Width + borderWidth;
                else form.Left = scn.WorkingArea.Right - form.Width;
                form.Top = scn.WorkingArea.Top;
            }
            else if (mpdnForm.Left == scn.WorkingArea.Width - mpdnForm.Width && mpdnForm.Height == scn.WorkingArea.Height)
            {
                int borderWidth = SystemInformation.SizingBorderWidth;

                docked = true;
                form.Width = scn.WorkingArea.Width / 2;
                form.Height = scn.WorkingArea.Height;
                if (Settings.LockWindowSize) form.Left = scn.WorkingArea.Left - borderWidth;
                else form.Left = scn.WorkingArea.Left;
                form.Top = scn.WorkingArea.Top;
            }
            else
            {
                if (!docked) return;
                docked = false;
                form.Size = formStartSize;

                if (Settings.LockWindowSize)
                {
                    int borderWidth = SystemInformation.SizingBorderWidth;
                    form.Left = mpdnForm.Right + borderWidth;
                    form.Top = mpdnForm.Top + borderWidth;
                }
                else
                {
                    form.Left = mpdnForm.Right;
                    form.Top = mpdnForm.Top;
                }
            }
        }

        public void SnapPlayer()
        {
            int borderWidth = SystemInformation.SizingBorderWidth;

            if (!resizing)
            {
                if (Settings.LockWindowSize) form.Height = mpdnForm.Height - (borderWidth * 2);
                else form.Height = mpdnForm.Height;
            }
            else
            {
                if (Settings.LockWindowSize)
                {
                    form.Width = mpdnForm.Width;
                    form.Height = mpdnForm.Height - (borderWidth * 2);
                }
                else form.Size = mpdnForm.Size;
            }

            if (Settings.LockWindowSize)
            {
                form.Left = mpdnForm.Right + borderWidth;
                form.Top = mpdnForm.Top + borderWidth;
            }
            else
            {
                form.Left = mpdnForm.Right;
                form.Top = mpdnForm.Top;
            }
        }

        private void OnMpdnFormResizeBegin(object sender, EventArgs e)
        {
            if (CursorIsOnResizeAnchor()) resizing = true;
            if (!docked) formStartSize = form.Size;
        }

        private void OnMpdnFormResizeEnd(object sender, EventArgs e)
        {
            resizing = false;
        }
        
        private bool CursorIsOnResizeAnchor()
        {
            if (Cursor.Current == Cursors.SizeNWSE || Cursor.Current == Cursors.SizeNESW) return true;
            else return false;
        }

        void OnFormSizeChanged(object sender, EventArgs e)
        {
            if (form.WindowState == FormWindowState.Minimized)
            {
                form.Bounds = form.RestoreBounds;
            }
        }

        public override IList<Verb> Verbs
        {
            get
            {
                return new[]
                {
                    new Verb(Category.File, string.Empty, "Open Playlist", "Ctrl+Alt+O", string.Empty, OpenPlaylist),
                    new Verb(Category.View, string.Empty, "Playlist", "Ctrl+Alt+P", string.Empty, ViewPlaylist),
                    new Verb(Category.Play, Subcategory, "Next", "Ctrl+Alt+N", string.Empty, () => form.PlayNext()),
                    new Verb(Category.Play, Subcategory, "Previous", "Ctrl+Alt+B", string.Empty, () => form.PlayPrevious())
                };
            }
        }

        public static bool IsPlaylistFile(string filename)
        {
            var extension = Path.GetExtension(filename);
            return extension != null && extension.ToLower() == ".mpl";
        }

        private void CloseMedia()
        {
            form.CloseMedia();
            form.SetPlayStyling();
        }

        private void CloseMPDN()
        {
            if (String.IsNullOrEmpty(PlayerControl.MediaFilePath)) return;
            if (PlayerControl.MediaPosition < PlayerControl.MediaDuration) return;
            if (form.GetDgvPlaylist().CurrentRow.Index < form.Playlist.Count - 1) return;
            mpdnForm.Close();
        }

        private void SetFormToSizable()
        {
            form.FormBorderStyle = FormBorderStyle.Sizable;
            StatusStrip s = (StatusStrip)form.Controls["statusStrip1"];
            s.SizingGrip = true;
        }

        private void SetFormToFixed()
        {
            form.FormBorderStyle = FormBorderStyle.FixedSingle;
            StatusStrip s = (StatusStrip)form.Controls["statusStrip1"];
            s.SizingGrip = false;
        }

        private void FixFormLocationBounds()
        {
            var screen = Screen.FromControl(mpdnForm);
            var screenBounds = screen.WorkingArea;
            var p = mpdnForm.PointToScreen(new Point(mpdnForm.Right, mpdnForm.Bottom));

            if (form.Left < 0)
                form.Left = 0;
            if (form.Left + form.Width > screenBounds.Width)
                form.Left = screenBounds.Width - form.Width;
            if (form.Top < 0)
                form.Top = 0;
            if (form.Top + form.Height > screenBounds.Height)
                form.Top = screenBounds.Height - form.Height;
        }

        private void SetActiveFile()
        {
            if (PlayerControl.PlayerState != PlayerState.Playing || form.Playlist.Count > 1) return;
            if (string.IsNullOrEmpty(PlayerControl.MediaFilePath)) return;

            if (form.CurrentItem != null && form.CurrentItem.FilePath != PlayerControl.MediaFilePath)
            {
                form.ActiveFile(PlayerControl.MediaFilePath);
            }
            else if (form.CurrentItem == null)
            {
                form.ActiveFile(PlayerControl.MediaFilePath);
            }
        }

        private void PlayNextInFolder()
        {
            if (PlayerControl.MediaPosition != PlayerControl.MediaDuration) return;
            form.PlayNextFileInDirectory();
        }

        private void AddFileToPlaylist()
        {
            if (string.IsNullOrEmpty(PlayerControl.MediaFilePath)) return;
            var foundFile = form.Playlist.Find(i => i.FilePath == PlayerControl.MediaFilePath);
            if (foundFile != null) return;
            form.AddActiveFile(PlayerControl.MediaFilePath);
        }

        private void NewPlaylist()
        {
            form.NewPlaylist();
        }

        private void OpenPlaylist()
        {
            form.Show(PlayerControl.VideoPanel);
            form.OpenPlaylist();
        }

        private void ViewPlaylist()
        {
            form.Show(PlayerControl.VideoPanel);
        }

        public string GetDirectoryName(string path)
        {
            if (path == null)
            {
                throw new ArgumentNullException("path");
            }

            return Path.GetDirectoryName(path) ?? Path.GetPathRoot(path);
        }

        public IEnumerable<string> GetMediaFiles(string mediaDir)
        {
            string[] filter = form.openFileDialog.Filter.Split('|');
            string[] extensions = filter[1].Replace(";","").Replace(" ", "").Split('*');
            var files = Directory.EnumerateFiles(mediaDir, "*.*", SearchOption.AllDirectories)
                .OrderBy(f => f, new PlayerExtensions.Playlist.NaturalSortComparer())
                .Where(file => extensions.Contains(Path.GetExtension(file.ToLower())));
            return files;
        }

        private void OnDragEnter(object sender, PlayerControlEventArgs<DragEventArgs> e)
        {
            e.Handled = true;
            e.InputArgs.Effect = DragDropEffects.Copy;
        }

        private void OnDragDrop(object sender, PlayerControlEventArgs<DragEventArgs> e)
        {
            var files = (string[])e.InputArgs.Data.GetData(DataFormats.FileDrop);

            if (files.Length == 1)
            {
                var filename = files[0];

                if (Directory.Exists(filename))
                {
                    var media = GetMediaFiles(filename);
                    form.CloseMedia();
                    form.ClearPlaylist();
                    form.AddFiles(media.ToArray());
                    form.SetPlaylistIndex(0);
                    return;
                }
                else if (IsPlaylistFile(filename))
                {
                    form.OpenPlaylist(filename);
                    return;
                }

                form.ActiveFile(filename);
                form.SetPlaylistIndex(0);
            }
            else
            {
                form.AddFiles(files);
                form.SetPlaylistIndex(0);
            }

            e.Handled = true;
        }

        private void OnCommandLineFileOpen(object sender, CommandLineFileOpenEventArgs e)
        {
            if (!IsPlaylistFile(e.Filename)) return;
            e.Handled = true;
            form.OpenPlaylist(e.Filename);
        }
    }

    public enum AfterPlaybackSettingsOpt
    {
        DoNothing = 0,
        PlayNextFileInFolder,
        ClosePlayer
    }

    public class PlaylistSettings
    {
        public bool ShowPlaylistOnStartup { get; set; }
        public AfterPlaybackSettingsOpt AfterPlaybackOpt { get; set; }
        public bool BeginPlaybackOnStartup { get; set; }
        public bool BeginPlaybackWhenFileIsAdded { get; set; }
        public bool BeginPlaybackWhenPlaylistFileIsOpened { get; set; }
        public bool AddToPlaylistOnFileOpen { get; set; }
        public bool RememberWindowSize { get; set; }
        public bool RememberWindowPosition { get; set; }
        public bool SnapAndScaleWithPlayer { get; set; }
        public bool RememberPlaylist { get; set; }
        public Point WindowPosition { get; set; }
        public Size WindowSize { get; set; }
        public bool LockWindowSize { get; set; }
        public bool RememberColumns { get; set; }
        public List<string> Columns { get; set; }
        public List<string> RememberedFiles { get; set; }

        public PlaylistSettings()
        {
            ShowPlaylistOnStartup = false;
            AfterPlaybackOpt = AfterPlaybackSettingsOpt.DoNothing;
            BeginPlaybackOnStartup = false;
            BeginPlaybackWhenFileIsAdded = false;
            BeginPlaybackWhenPlaylistFileIsOpened = false;
            AddToPlaylistOnFileOpen = false;
            SnapAndScaleWithPlayer = false;
            RememberColumns = false;
            RememberWindowPosition = false;
            RememberWindowSize = false;
            LockWindowSize = false;
            RememberPlaylist = false;
            Columns = new List<string>();
            RememberedFiles = new List<string>();
        }
    }
}
