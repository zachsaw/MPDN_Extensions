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
        #region Fields

        private const string Subcategory = "Playlist";

        private readonly PlaylistForm form = new PlaylistForm();
        private readonly PlayerMenuItem menuItem = new PlayerMenuItem();

        private bool docked;

        private Form mpdnForm;
        private Point mpdnStartLocation;

        private Point formStartLocation;
        private Size formStartSize;
        private bool moving;
        private bool resizing;

        #endregion

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

        public override IList<Verb> Verbs
        {
            get
            {
                return new[]
                {
                    new Verb(Category.File, string.Empty, "Open Playlist", "Ctrl+Alt+O", string.Empty, OpenPlaylist),
                    new Verb(Category.View, string.Empty, "Playlist", "Ctrl+Alt+P", string.Empty, ViewPlaylist, menuItem),
                    new Verb(Category.Play, Subcategory, "Next", "Ctrl+Alt+N", string.Empty, () => form.PlayNext()),
                    new Verb(Category.Play, Subcategory, "Previous", "Ctrl+Alt+B", string.Empty, () => form.PlayPrevious())
                };
            }
        }

        #region Playlist (re)init and dispose

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

            if (Settings.SnapWithPlayer)
            {
                form.SnapWithPlayer = Settings.SnapWithPlayer;
                SnapPlayer();
            }

            if (Settings.StaySnapped)
            {
                form.KeepSnapped = Settings.StaySnapped;
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

            if (Settings.RememberPlaylist)
            {
                if (Settings.RememberedFiles.Count > 0)
                {
                    var playList = new List<PlaylistItem>();

                    foreach (var f in Settings.RememberedFiles)
                    {
                        var s = f.Split('|');
                        string filePath = s[0];
                        var skipChapters = new List<int>();
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
                        string duration = s[4];

                        playList.Add(new PlaylistItem(filePath, skipChapters, endChapter, active, duration));
                    }

                    form.Playlist = playList;
                    form.PopulatePlaylist();
                    form.RefreshPlaylist();

                    if (Settings.BeginPlaybackOnStartup)
                    {
                        form.PlayActive();
                    }
                }
            }

            //FixFormLocationBounds();
        }

        public override void Destroy()
        {
            PlayerControl.PlayerStateChanged -= OnPlayerStateChanged;
            PlayerControl.PlaybackCompleted -= OnPlaybackCompleted;
            PlayerControl.FormClosed -= OnMpdnFormClosed;
            PlayerControl.DragEnter -= OnDragEnter;
            PlayerControl.DragDrop -= OnDragDrop;
            PlayerControl.CommandLineFileOpen -= OnCommandLineFileOpen;
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

        public void Reinitialize()
        {
            if (Settings.LockWindowSize) SetFormToFixed();
            else SetFormToSizable();
            if (Settings.SnapWithPlayer) SnapPlayer();

            form.RememberWindowPosition = Settings.RememberWindowPosition;
            form.RememberWindowSize = Settings.RememberWindowSize;
            form.SnapWithPlayer = Settings.SnapWithPlayer;
            form.KeepSnapped = Settings.StaySnapped;
            form.LockWindowSize = Settings.LockWindowSize;
            form.BeginPlaybackOnStartup = Settings.BeginPlaybackOnStartup;
        }

        public PlaylistForm GetPlaylistForm
        {
            get { return form; }
        }

        #endregion

        #region The Methods

        public void ViewPlaylist()
        {
            if (form.Visible)
                form.Hide();
            else
                form.Show(PlayerControl.VideoPanel);
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

        public IEnumerable<string> GetAllMediaFiles(string mediaDir)
        {
            var filter = form.openFileDialog.Filter.Split('|');
            var extensions = filter[1].Replace(";", "").Replace(" ", "").Split('*');

            var files = Directory.EnumerateFiles(mediaDir, "*.*", SearchOption.AllDirectories)
            .OrderBy(Path.GetDirectoryName, new NaturalSortComparer())
            .ThenBy(Path.GetFileName, new NaturalSortComparer())
            .Where(Path.HasExtension)
            .Where(f => extensions.Contains(Path.GetExtension(f.ToLower())));

            return files;
        }

        public IEnumerable<string> GetMediaFiles(string mediaDir)
        {
            var filter = form.openFileDialog.Filter.Split('|');
            var extensions = filter[1].Replace(";", "").Replace(" ", "").Split('*');

            var files = Directory.EnumerateFiles(mediaDir, "*.*", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetDirectoryName, new NaturalSortComparer())
            .ThenBy(Path.GetFileName, new NaturalSortComparer())
            .Where(Path.HasExtension)
            .Where(f => extensions.Contains(Path.GetExtension(f.ToLower())));

            return files;
        }

        private void RememberSettings()
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
                if (form.Playlist.Count <= 0) return;

                foreach (var i in form.Playlist)
                {
                    string skipChapters = "";

                    if (i.SkipChapters != null && i.SkipChapters.Count > 0)
                    {
                        skipChapters = string.Join(",", i.SkipChapters);
                    }

                    Settings.RememberedFiles.Add(i.FilePath + "|" + skipChapters + "|" + i.EndChapter + "|" +
                                                 i.Active + "|" + i.Duration);
                }
            }
        }

        public void SnapPlayer()
        {
            int borderWidth = SystemInformation.SizingBorderWidth;

            if (!resizing)
            {
                if (Settings.ScaleWithPlayer)
                {
                    if (Settings.LockWindowSize) form.Height = mpdnForm.Height - (borderWidth * 2);
                    else form.Height = mpdnForm.Height;
                }
            }
            else
            {
                if (Settings.ScaleWithPlayer)
                {
                    if (Settings.LockWindowSize)
                    {
                        form.Width = mpdnForm.Width;
                        form.Height = mpdnForm.Height - (borderWidth * 2);
                    }
                    else form.Size = mpdnForm.Size;
                }
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

        #endregion

        #region Helper Methods

        private void FixFormLocationBounds()
        {
            var screen = Screen.FromControl(mpdnForm);
            var screenBounds = screen.WorkingArea;
            if (form.Left < 0)
                form.Left = 0;
            if (form.Left + form.Width > screenBounds.Width)
                form.Left = screenBounds.Width - form.Width;
            if (form.Top < 0)
                form.Top = 0;
            if (form.Top + form.Height > screenBounds.Height)
                form.Top = screenBounds.Height - form.Height;
        }

        private void SetFormToSizable()
        {
            form.FormBorderStyle = FormBorderStyle.Sizable;
            var s = (StatusStrip)form.Controls["statusStrip1"];
            s.SizingGrip = true;
        }

        private void SetFormToFixed()
        {
            form.FormBorderStyle = FormBorderStyle.FixedSingle;
            var s = (StatusStrip)form.Controls["statusStrip1"];
            s.SizingGrip = false;
        }

        private void CloseMedia()
        {
            form.CloseMedia();
            form.SetPlayStyling();
        }

        private void CloseMpdn()
        {
            if (String.IsNullOrEmpty(PlayerControl.MediaFilePath)) return;
            if (PlayerControl.MediaPosition < PlayerControl.MediaDuration) return;
            var row = form.GetDgvPlaylist().CurrentRow;
            if (row == null) return;
            if (row.Index < form.Playlist.Count - 1) return;
            mpdnForm.Close();
        }

        public static bool IsPlaylistFile(string filename)
        {
            var extension = Path.GetExtension(filename);
            return extension != null && extension.ToLower() == ".mpl";
        }

        private bool IsOnScreen()
        {
            return (Screen.AllScreens.Select(
                    screen => new { screen, formRectangle = new Rectangle(form.Left, form.Top, form.Width, form.Height) })
                    .Where(t => t.screen.WorkingArea.Contains(t.formRectangle))
                    .Select(t => t.screen)).Any();
        }

        private static bool CursorIsOnResizeAnchor()
        {
            return Cursor.Current == Cursors.SizeNWSE || Cursor.Current == Cursors.SizeNESW;
        }

        #endregion

        #region PlayerControl Events

        private void OnFormCloseMedia(object sender, EventArgs e)
        {
            CloseMedia();
        }

        private void OnPlayerStateChanged(object sender, EventArgs e)
        {
            SetActiveFile();
        }

        private void OnPlaybackCompleted(object sender, EventArgs e)
        {
            switch (Settings.AfterPlaybackOpt)
            {
                case AfterPlaybackSettingsOpt.PlayNextFileInFolder:
                    PlayNextInFolder();
                    break;
                case AfterPlaybackSettingsOpt.ClosePlayer:
                    CloseMpdn();
                    break;
            }
        }

        private void OnFormVisibilityChanged(object sender, EventArgs e)
        {
            mpdnForm.BringToFront();
            menuItem.Checked = form.Visible;
        }

        private void OnFormMove(object sender, EventArgs e)
        {
            if (!IsOnScreen()) form.GetDgvPlaylist().Invalidate();

            if (moving)
                return;

            if (mpdnForm.WindowState == FormWindowState.Minimized && form.WindowState == FormWindowState.Minimized)
                return;
            mpdnStartLocation = mpdnForm.Location;
            formStartLocation = form.Location;
        }

        private void OnFormSizeChanged(object sender, EventArgs e)
        {
            if (form.WindowState == FormWindowState.Minimized)
            {
                form.Bounds = form.RestoreBounds;
            }
        }

        private void OnDragEnter(object sender, PlayerControlEventArgs<DragEventArgs> e)
        {
            e.Handled = true;
            e.InputArgs.Effect = DragDropEffects.Copy;
        }

        private void OnDragDrop(object sender, PlayerControlEventArgs<DragEventArgs> e)
        {
            var filter = form.openFileDialog.Filter.Split('|');
            var extensions = filter[1].Replace(";", "").Replace(" ", "").Split('*');

            var files = (string[])e.InputArgs.Data.GetData(DataFormats.FileDrop);
            if (files == null)
                return;

            if (files.Length == 1)
            {
                var filename = files[0];

                if (Directory.Exists(filename))
                {
                    var media = GetAllMediaFiles(filename);
                    form.CloseMedia();
                    form.ClearPlaylist();
                    form.AddFiles(
                        media.Where(file => extensions.Contains(PathHelper.GetExtension(file.ToLower())))
                            .OrderBy(f => f, new NaturalSortComparer())
                            .Where(f => PathHelper.GetExtension(f).Length > 0)
                            .ToArray());
                    form.SetPlaylistIndex(0);
                    return;
                }

                if (IsPlaylistFile(filename))
                {
                    form.OpenPlaylist(filename);
                    return;
                }

                if (PathHelper.GetExtension(filename).Length < 1 || !extensions.Contains(Path.GetExtension(filename))) return;

                form.ActiveFile(filename);
                form.SetPlaylistIndex(0);
            }
            else
            {
                form.AddFiles(
                    files.Where(file => extensions.Contains(PathHelper.GetExtension(file.ToLower())))
                        .OrderBy(f => f, new NaturalSortComparer())
                        .Where(f => PathHelper.GetExtension(f).Length > 0)
                        .ToArray());
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

        #endregion

        #region MPDN Form Events

        private void OnMpdnFormClosed(object sender, EventArgs e)
        {
            RememberSettings();
        }

        private void OnMpdnFormMove(object sender, EventArgs e)
        {
            if (form.WindowState == FormWindowState.Minimized) return;
            if (CursorIsOnResizeAnchor()) return;

            moving = true;

            if (Settings.SnapWithPlayer)
            {
                form.Left = formStartLocation.X + mpdnForm.Location.X - mpdnStartLocation.X;
                form.Top = formStartLocation.Y + mpdnForm.Location.Y - mpdnStartLocation.Y;
            }

            moving = false;
        }

        void OnMpdnFormSizeChanged(object sender, EventArgs e)
        {
            var scn = Screen.FromPoint(mpdnForm.Location);

            if (Settings.SnapWithPlayer) SnapPlayer();

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

        private void OnMpdnFormResizeBegin(object sender, EventArgs e)
        {
            if (CursorIsOnResizeAnchor()) resizing = true;
            if (!docked) formStartSize = form.Size;
        }

        private void OnMpdnFormResizeEnd(object sender, EventArgs e)
        {
            resizing = false;
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
                    Cursor.Position = new Point(form.Location.X + 100, form.Location.Y + 100);
                    e.SuppressKeyPress = true;
                }
            }
        }

        private void OnMpdnFormOpenClick(object sender, EventArgs e)
        {
            NewPlaylist();
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

        #endregion
    }

    #region AfterPlaybackSettingsOpt

    public enum AfterPlaybackSettingsOpt
    {
        DoNothing = 0,
        PlayNextFileInFolder,
        ClosePlayer
    }

    #endregion

    #region PlaylistSettings

    public class PlaylistSettings
    {
        public bool ShowPlaylistOnStartup { get; set; }
        public AfterPlaybackSettingsOpt AfterPlaybackOpt { get; set; }
        public bool BeginPlaybackOnStartup { get; set; }
        public bool RememberWindowSize { get; set; }
        public bool RememberWindowPosition { get; set; }
        public bool SnapWithPlayer { get; set; }
        public bool ScaleWithPlayer { get; set; }
        public bool StaySnapped { get; set; }
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
            SnapWithPlayer = true;
            StaySnapped = false;
            RememberColumns = false;
            RememberWindowPosition = false;
            RememberWindowSize = false;
            LockWindowSize = false;
            RememberPlaylist = false;
            Columns = new List<string>();
            RememberedFiles = new List<string>();
        }
    }

    #endregion
}
