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
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Windows.Forms;
using Mpdn.Extensions.Framework;

namespace Mpdn.Extensions.PlayerExtensions.Playlist
{
    public class Playlist : PlayerExtension<PlaylistSettings, PlaylistConfigDialog>
    {
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
                    new Verb(Category.View, string.Empty, "Playlist", "Ctrl+Alt+P", string.Empty, ViewPlaylist, m_MenuItem),
                    new Verb(Category.Play, SUBCATEGORY, "Next", "Ctrl+Alt+N", string.Empty, () => m_Form.PlayNext()),
                    new Verb(Category.Play, SUBCATEGORY, "Previous", "Ctrl+Alt+B", string.Empty,
                        () => m_Form.PlayPrevious())
                };
            }
        }

        #region Fields

        private const string SUBCATEGORY = "Playlist";
        private const int ICON_BASE_SIZE = 16;
        private string OS_VERSION = Environment.OSVersion.ToString();

        private readonly PlaylistForm m_Form = new PlaylistForm();
        private readonly PlayerMenuItem m_MenuItem = new PlayerMenuItem();

        private bool m_Docked;

        private Form m_MpdnForm;
        private Point m_MpdnStartLocation;

        private Point m_FormStartLocation;
        private Size m_FormStartSize;
        private bool m_Moving;
        private bool m_Resizing;

        #endregion

        #region Playlist (re)init and dispose

        public override void Initialize()
        {
            base.Initialize();

            PlaylistForm.IconSize = Convert.ToInt32((ICON_BASE_SIZE / 100f) * int.Parse(Regex.Match(Settings.IconScale.ToString(), @"\d+").Value));
            if (!string.IsNullOrEmpty(Settings.Theme)) m_Form.Theme = Settings.Theme;

            m_Form.Setup(this);

            Player.StateChanged += OnPlayerStateChanged;
            Player.Playback.Completed += OnPlaybackCompleted;
            Player.Closed += OnMpdnFormClosed;
            Player.DragEnter += OnDragEnter;
            Player.DragDrop += OnDragDrop;
            Player.CommandLineFileOpen += OnCommandLineFileOpen;
            m_MpdnForm = Player.ActiveForm;
            m_MpdnForm.Move += OnMpdnFormMove;
            m_MpdnForm.KeyDown += OnMpdnFormKeyDown;
            m_MpdnForm.MainMenuStrip.MenuActivate += OnMpdnFormMainMenuActivated;
            m_MpdnForm.SizeChanged += OnMpdnFormSizeChanged;
            m_MpdnForm.ResizeBegin += OnMpdnFormResizeBegin;
            m_MpdnForm.ResizeEnd += OnMpdnFormResizeEnd;
            m_Form.VisibleChanged += OnFormVisibilityChanged;
            m_Form.Move += OnFormMove;
            m_Form.SizeChanged += OnFormSizeChanged;

            if (Settings.RememberWindowPosition)
            {
                m_Form.RememberWindowPosition = Settings.RememberWindowPosition;
                m_Form.WindowPosition = Settings.WindowPosition;
            }

            if (Settings.RememberWindowSize)
            {
                m_Form.RememberWindowSize = Settings.RememberWindowSize;
                m_Form.WindowSize = Settings.WindowSize;
                m_FormStartSize = m_Form.Size;
            }

            if (Settings.LockWindowSize)
            {
                m_Form.LockWindowSize = Settings.LockWindowSize;
                SetFormToFixed();
            }

            if (Settings.SnapWithPlayer)
            {
                m_Form.SnapWithPlayer = Settings.SnapWithPlayer;
                SnapPlayer();
            }

            if (Settings.StaySnapped) m_Form.KeepSnapped = Settings.StaySnapped;

            if (Settings.RememberColumns) if (Settings.Columns != null && Settings.Columns.Count > 0) m_Form.Columns = Settings.Columns;

            if (Settings.ShowToolTips) m_Form.ShowToolTips = Settings.ShowToolTips;

            if (Settings.ShowPlaylistOnStartup) ViewPlaylist();

            if (Settings.BeginPlaybackOnStartup) m_Form.BeginPlaybackOnStartup = Settings.BeginPlaybackOnStartup;

            if (Settings.RegexList != null && Settings.RegexList.Count > 0) m_Form.RegexList = Settings.RegexList;

            if (Settings.StripDirectoryInFileName) m_Form.StripDirectoryInFileName = Settings.StripDirectoryInFileName;

            m_Form.AfterPlaybackAction = Settings.AfterPlaybackAction;

            if (Settings.RememberPlaylist)
            {
                if (Settings.RememberedFiles.Count > 0)
                {
                    var playList = new List<PlaylistItem>();

                    foreach (string f in Settings.RememberedFiles)
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
                        string duration = s[4];

                        playList.Add(new PlaylistItem(filePath, skipChapters, endChapter, active, duration));
                    }

                    m_Form.Playlist = playList;
                    m_Form.PopulatePlaylist();
                    m_Form.RefreshPlaylist();
                    Task.Factory.StartNew(m_Form.GetMediaDuration);

                    if (Settings.BeginPlaybackOnStartup) m_Form.PlayActive();
                }
            }

            FixFormLocationBounds();
            BindContextMenu(m_MpdnForm);
        }

        public override void Destroy()
        {
            Player.StateChanged -= OnPlayerStateChanged;
            Player.Playback.Completed -= OnPlaybackCompleted;
            Player.Closed -= OnMpdnFormClosed;
            Player.DragEnter -= OnDragEnter;
            Player.DragDrop -= OnDragDrop;
            Player.CommandLineFileOpen -= OnCommandLineFileOpen;
            m_MpdnForm.Move -= OnMpdnFormMove;
            m_MpdnForm.KeyDown -= OnMpdnFormKeyDown;
            m_MpdnForm.MainMenuStrip.MenuActivate -= OnMpdnFormMainMenuActivated;
            m_MpdnForm.SizeChanged -= OnMpdnFormSizeChanged;
            m_MpdnForm.ResizeBegin -= OnMpdnFormResizeBegin;
            m_MpdnForm.ResizeEnd -= OnMpdnFormResizeEnd;
            m_Form.VisibleChanged -= OnFormVisibilityChanged;
            m_Form.Move -= OnFormMove;
            m_Form.SizeChanged -= OnFormSizeChanged;

            base.Destroy();
            m_Form.Dispose();
        }

        public void Reinitialize()
        {
            PlaylistForm.IconSize = Convert.ToInt32((ICON_BASE_SIZE / 100f) * int.Parse(Regex.Match(Settings.IconScale.ToString(), @"\d+").Value));
            m_Form.Theme = Settings.Theme;
            m_Form.LoadCustomSettings();

            if (Settings.LockWindowSize) SetFormToFixed();
            else SetFormToSizable();
            if (Settings.SnapWithPlayer) SnapPlayer();

            m_Form.RememberWindowPosition = Settings.RememberWindowPosition;
            m_Form.RememberWindowSize = Settings.RememberWindowSize;
            m_Form.SnapWithPlayer = Settings.SnapWithPlayer;
            m_Form.KeepSnapped = Settings.StaySnapped;
            m_Form.LockWindowSize = Settings.LockWindowSize;
            m_Form.BeginPlaybackOnStartup = Settings.BeginPlaybackOnStartup;
            m_Form.AfterPlaybackAction = Settings.AfterPlaybackAction;
            m_Form.ShowToolTips = Settings.ShowToolTips;

            m_Form.SetControlStates();
        }

        public void SyncSettings()
        {
            Settings.RegexList = m_Form.RegexList;
        }

        public PlaylistForm GetPlaylistForm
        {
            get { return m_Form; }
        }

        #endregion

        #region The Methods

        public void ViewPlaylist()
        {
            if (m_Form.Visible) m_Form.Hide();
            else m_Form.Show(Gui.VideoBox);
        }

        private void NewPlaylist()
        {
            m_Form.NewPlaylist();
        }

        private void OpenPlaylist()
        {
            m_Form.Show(Gui.VideoBox);
            m_Form.OpenPlaylist();
        }

        private void SetActiveFile()
        {
            if (String.IsNullOrEmpty(Media.FilePath)) return;
            if (!File.Exists(Media.FilePath)) return;
            if (Player.State != PlayerState.Playing) return;

            if (m_Form.CurrentItem != null && m_Form.CurrentItem.FilePath != Media.FilePath) m_Form.ActiveFile(Media.FilePath);
            else if (m_Form.CurrentItem == null) m_Form.ActiveFile(Media.FilePath);
        }

        private void PlayNextInFolder()
        {
            if (Media.Position != Media.Duration) return;
            m_Form.PlayNextFileInDirectory();
        }

        private void RepeatPlaylist()
        {
            var lastItem = m_Form.Playlist.Last();
            if (m_Form.CurrentItem != lastItem) return;
            if (Media.Position != Media.Duration) return;
            m_Form.ResetPlayCount();
            m_Form.SetPlaylistIndex(0);
        }

        public IEnumerable<string> GetAllMediaFiles(string mediaDir)
        {
            var filter = m_Form.openFileDialog.Filter.Split('|');
            var extensions = filter[1].Replace(";", string.Empty).Replace(" ", string.Empty).Split('*');

            var files = Directory.EnumerateFiles(mediaDir, "*.*", SearchOption.AllDirectories)
                .OrderBy(Path.GetDirectoryName, new NaturalSortComparer())
                .ThenBy(Path.GetFileName, new NaturalSortComparer())
                .Where(Path.HasExtension)
                .Where(f => extensions.Contains(Path.GetExtension(f.ToLower())));

            return files;
        }

        public IEnumerable<string> GetMediaFiles(string mediaDir)
        {
            var filter = m_Form.openFileDialog.Filter.Split('|');
            var extensions = filter[1].Replace(";", string.Empty).Replace(" ", string.Empty).Split('*');

            var files = Directory.EnumerateFiles(mediaDir, "*.*", SearchOption.TopDirectoryOnly)
                .OrderBy(Path.GetDirectoryName, new NaturalSortComparer())
                .ThenBy(Path.GetFileName, new NaturalSortComparer())
                .Where(Path.HasExtension)
                .Where(f => extensions.Contains(Path.GetExtension(f.ToLower())));

            return files;
        }

        private void RememberSettings()
        {
            Settings.WindowPosition = m_Form.Location;
            Settings.WindowSize = m_Form.Size;

            Settings.Columns.Clear();

            for (var i = 0; i < m_Form.GetDgvPlaylist().Columns.Count; i++)
            {
                var c = m_Form.GetDgvPlaylist().Columns[i];
                Settings.Columns.Add(c.Name + "|"
                                     + c.Visible + "|" + c.Width);
            }

            if (Settings.RememberPlaylist)
            {
                Settings.RememberedFiles.Clear();
                if (m_Form.Playlist.Count == 0) return;

                foreach (var i in m_Form.Playlist)
                {
                    string skipChapters = string.Empty;

                    if (i.SkipChapters != null && i.SkipChapters.Count > 0) skipChapters = string.Join(",", i.SkipChapters);

                    Settings.RememberedFiles.Add(i.FilePath + "|" + skipChapters + "|" + i.EndChapter + "|" +
                                                 i.Active + "|" + i.Duration);
                }
            }

            if (m_Form.RegexList != null && m_Form.RegexList.Count > 0) Settings.RegexList = m_Form.RegexList;
        }

        public void SnapPlayer()
        {
            int borderWidth = SystemInformation.SizingBorderWidth;

            if (!m_Resizing)
            {
                if (Settings.ScaleWithPlayer)
                {
                    if (Settings.LockWindowSize)
                    {
                        if (OS_VERSION.Contains("6.3")) //check if OS is Windows 10
                        {
                            m_Form.Height = m_MpdnForm.Height + borderWidth;
                        }
                        else
                        {
                            m_Form.Height = m_MpdnForm.Height - borderWidth * 2;
                        }
                    }
                    else m_Form.Height = m_MpdnForm.Height;
                }
            }
            else
            {
                if (Settings.ScaleWithPlayer)
                {
                    if (Settings.LockWindowSize)
                    {
                        if (OS_VERSION.Contains("6.3")) //check if OS is Windows 10
                        {
                            m_Form.Width = m_MpdnForm.Width;
                            m_Form.Height = m_MpdnForm.Height - borderWidth;
                        }
                        else
                        {
                            m_Form.Width = m_MpdnForm.Width;
                            m_Form.Height = m_MpdnForm.Height - borderWidth * 2;
                        }
                    }
                    else m_Form.Size = m_MpdnForm.Size;
                }
            }

            if (Settings.LockWindowSize)
            {
                if (OS_VERSION.Contains("6.3")) //check if OS is Windows 10
                {
                    m_Form.Left = m_MpdnForm.Right - (borderWidth * 2);
                    m_Form.Top = m_MpdnForm.Top;
                }
                else
                {
                    m_Form.Left = m_MpdnForm.Right + borderWidth;
                    m_Form.Top = m_MpdnForm.Top + borderWidth;
                }
            }
            else
            {
                if (OS_VERSION.Contains("6.3")) //check if OS is Windows 10
                {
                    m_Form.Left = m_MpdnForm.Right - (borderWidth * 2) - 5;
                    m_Form.Top = m_MpdnForm.Top;
                }
                else
                {
                    m_Form.Left = m_MpdnForm.Right;
                    m_Form.Top = m_MpdnForm.Top;
                }
            }
        }

        #endregion

        #region Helper Methods

        private void BindContextMenu(Control ctrl)
        {
            foreach (Control c in ctrl.Controls)
            {
                if (c.ContextMenuStrip != null) c.ContextMenuStrip.Opened += OnMpdnFormContextMenuOpened;

                if (c.Controls.Count > 0) BindContextMenu(c);
            }
        }

        private void FixFormLocationBounds()
        {
            var screen = Screen.FromControl(m_MpdnForm);
            var screenBounds = screen.WorkingArea;
            if (m_Form.Left < 0) m_Form.Left = 0;
            if (m_Form.Left > screenBounds.Width) m_Form.Left = screenBounds.Width - m_Form.Width;
            if (m_Form.Top < 0) m_Form.Top = 0;
            if (m_Form.Top > screenBounds.Height) m_Form.Top = screenBounds.Height - m_Form.Height;
        }

        private void SetFormToSizable()
        {
            m_Form.FormBorderStyle = FormBorderStyle.Sizable;
            var s = (StatusStrip)m_Form.Controls["statusStrip1"];
            s.SizingGrip = true;
        }

        private void SetFormToFixed()
        {
            m_Form.FormBorderStyle = FormBorderStyle.FixedSingle;
            var s = (StatusStrip)m_Form.Controls["statusStrip1"];
            s.SizingGrip = false;
        }

        private void CloseMedia()
        {
            m_Form.CloseMedia();
            m_Form.SetPlayStyling();
        }

        private void CloseMpdn()
        {
            if (String.IsNullOrEmpty(Media.FilePath)) return;
            if (Media.Position < Media.Duration) return;

            var row = m_Form.GetDgvPlaylist().CurrentRow;

            if (row == null) return;
            if (row.Index < m_Form.Playlist.Count - 1) return;

            m_MpdnForm.Close();
        }

        public static bool IsPlaylistFile(string filename)
        {
            string extension = Path.GetExtension(filename);
            return extension != null && extension.ToLower() == ".mpl";
        }

        private bool IsOnScreen()
        {
            return (Screen.AllScreens.Select(
                screen => new {screen, formRectangle = new Rectangle(m_Form.Left, m_Form.Top, m_Form.Width, m_Form.Height)})
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
                case AfterPlaybackSettingsOpt.ClosePlayer:
                    CloseMpdn();
                    break;
                case AfterPlaybackSettingsOpt.PlayNextFileInFolder:
                    PlayNextInFolder();
                    break;
                case AfterPlaybackSettingsOpt.RepeatPlaylist:
                    RepeatPlaylist();
                    break;
            }
        }

        private void OnFormVisibilityChanged(object sender, EventArgs e)
        {
            m_MpdnForm.BringToFront();
            m_MenuItem.Checked = m_Form.Visible;
        }

        private void OnFormMove(object sender, EventArgs e)
        {
            if (!IsOnScreen()) m_Form.GetDgvPlaylist().Invalidate();

            if (m_Moving) return;

            if (m_MpdnForm.WindowState == FormWindowState.Minimized && m_Form.WindowState == FormWindowState.Minimized) return;

            m_MpdnStartLocation = m_MpdnForm.Location;
            m_FormStartLocation = m_Form.Location;
        }

        private void OnFormSizeChanged(object sender, EventArgs e)
        {
            if (m_Form.WindowState == FormWindowState.Minimized) m_Form.Bounds = m_Form.RestoreBounds;
        }

        private void OnDragEnter(object sender, PlayerControlEventArgs<DragEventArgs> e)
        {
            e.Handled = true;
            e.InputArgs.Effect = DragDropEffects.Copy;
        }

        private void OnDragDrop(object sender, PlayerControlEventArgs<DragEventArgs> e)
        {
            var filter = m_Form.openFileDialog.Filter.Split('|');
            var extensions = filter[1].Replace(";", string.Empty).Replace(" ", string.Empty).Split('*');

            var files = (string[])e.InputArgs.Data.GetData(DataFormats.FileDrop);
            if (files == null) return;

            if (files.Length == 1)
            {
                string filename = files[0];

                if (Directory.Exists(filename))
                {
                    var media = GetAllMediaFiles(filename);
                    m_Form.CloseMedia();
                    m_Form.ClearPlaylist();
                    m_Form.AddFiles(
                        media.Where(file => extensions.Contains(PathHelper.GetExtension(file.ToLower())))
                            .OrderBy(f => f, new NaturalSortComparer())
                            .Where(f => PathHelper.GetExtension(f).Length > 0)
                            .ToArray());
                    m_Form.SetPlaylistIndex(0);
                    return;
                }

                if (IsPlaylistFile(filename))
                {
                    m_Form.OpenPlaylist(filename);
                    return;
                }

                if (PathHelper.GetExtension(filename).Length < 1 || !extensions.Contains(Path.GetExtension(filename))) return;

                m_Form.ActiveFile(filename);
                m_Form.SetPlaylistIndex(0);
            }
            else
            {
                m_Form.AddFiles(
                    files.Where(file => extensions.Contains(PathHelper.GetExtension(file.ToLower())))
                        .OrderBy(f => f, new NaturalSortComparer())
                        .Where(f => PathHelper.GetExtension(f).Length > 0)
                        .ToArray());
                m_Form.SetPlaylistIndex(0);
            }

            e.Handled = true;
        }

        private void OnCommandLineFileOpen(object sender, CommandLineFileOpenEventArgs e)
        {
            if (!IsPlaylistFile(e.Filename)) return;
            e.Handled = true;
            m_Form.OpenPlaylist(e.Filename);
        }

        #endregion

        #region MPDN Form Events

        private void OnMpdnFormClosed(object sender, EventArgs e)
        {
            RememberSettings();
        }

        private void OnMpdnFormMove(object sender, EventArgs e)
        {
            if (m_Form.WindowState == FormWindowState.Minimized) return;
            if (CursorIsOnResizeAnchor()) return;

            m_Moving = true;

            if (Settings.SnapWithPlayer)
            {
                m_Form.Left = m_FormStartLocation.X + m_MpdnForm.Location.X - m_MpdnStartLocation.X;
                m_Form.Top = m_FormStartLocation.Y + m_MpdnForm.Location.Y - m_MpdnStartLocation.Y;
            }

            m_Moving = false;
        }

        private void OnMpdnFormSizeChanged(object sender, EventArgs e)
        {
            var scn = Screen.FromPoint(m_MpdnForm.Location);

            if (Settings.SnapWithPlayer) SnapPlayer();

            if (m_MpdnForm.WindowState == FormWindowState.Minimized) m_Form.Bounds = m_Form.RestoreBounds;

            if (m_MpdnForm.Left == 0 && m_MpdnForm.Height == scn.WorkingArea.Height && !CursorIsOnResizeAnchor())
            {
                int borderWidth = SystemInformation.SizingBorderWidth;

                m_Docked = true;
                m_Form.Width = scn.WorkingArea.Width / 2;
                m_Form.Height = scn.WorkingArea.Height;
                if (Settings.LockWindowSize) m_Form.Left = scn.WorkingArea.Right - m_Form.Width + borderWidth;
                else m_Form.Left = scn.WorkingArea.Right - m_Form.Width;
                m_Form.Top = scn.WorkingArea.Top;
            }
            else if (m_MpdnForm.Left == scn.WorkingArea.Width - m_MpdnForm.Width && m_MpdnForm.Height == scn.WorkingArea.Height && !CursorIsOnResizeAnchor())
            {
                int borderWidth = SystemInformation.SizingBorderWidth;

                m_Docked = true;
                m_Form.Width = scn.WorkingArea.Width / 2;
                m_Form.Height = scn.WorkingArea.Height;
                if (Settings.LockWindowSize) m_Form.Left = scn.WorkingArea.Left - borderWidth;
                else m_Form.Left = scn.WorkingArea.Left;
                m_Form.Top = scn.WorkingArea.Top;
            }
            else
            {
                if (!m_Docked) return;
                m_Docked = false;
                m_Form.Size = m_FormStartSize;

                if (Settings.LockWindowSize)
                {
                    int borderWidth = SystemInformation.SizingBorderWidth;
                    m_Form.Left = m_MpdnForm.Right + borderWidth;
                    m_Form.Top = m_MpdnForm.Top + borderWidth;
                }
                else
                {
                    m_Form.Left = m_Form.Left < m_MpdnForm.Left ? m_MpdnForm.Left - m_Form.Width : m_MpdnForm.Right;
                    m_Form.Top = m_MpdnForm.Top;
                }
            }
        }

        private void OnMpdnFormResizeBegin(object sender, EventArgs e)
        {
            if (CursorIsOnResizeAnchor()) m_Resizing = true;
            if (!m_Docked) m_FormStartSize = m_Form.Size;
        }

        private void OnMpdnFormResizeEnd(object sender, EventArgs e)
        {
            m_Resizing = false;
        }

        private void OnMpdnFormKeyDown(object sender, KeyEventArgs e)
        {
            if (e.Modifiers == Keys.Control && e.KeyCode == Keys.Tab)
            {
                if (Player.FullScreenMode.Active || !m_Form.Visible || m_Form.ContainsFocus) return;
                m_Form.Activate();
                Cursor.Position = new Point(m_Form.Location.X + 100, m_Form.Location.Y + 100);
                e.SuppressKeyPress = true;
            }
        }

        private void OnMpdnFormOpenClick(object sender, EventArgs e)
        {
            NewPlaylist();
        }

        private void OnMpdnFormMainMenuActivated(object sender, EventArgs e)
        {
            foreach (ToolStripMenuItem item in m_MpdnForm.MainMenuStrip.Items)
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

        private void OnMpdnFormContextMenuOpened(object sender, EventArgs e)
        {
            var s = sender as ContextMenuStrip;

            foreach (var item in s.Items.OfType<ToolStripMenuItem>().Where(item => item.Name == "menuFile"))
            {
                if (item.DropDownItems[0].Name == "menuQuickOpen")
                {
                    item.DropDownItems[0].Click -= OnMpdnFormOpenClick;
                    item.DropDownItems[0].Click += OnMpdnFormOpenClick;
                }

                if (item.DropDownItems[1].Name == "openToolStripMenuItem1")
                {
                    item.DropDownItems[1].Click -= OnMpdnFormOpenClick;
                    item.DropDownItems[1].Click += OnMpdnFormOpenClick;
                }

                if (item.DropDownItems[2].Name == "menuClose")
                {
                    item.DropDownItems[2].Click -= OnFormCloseMedia;
                    item.DropDownItems[2].Click += OnFormCloseMedia;
                }
            }
        }

        #endregion
    }

    #region Enums

    public enum IconScale
    {
        Scale100X = 0,
        Scale125X,
        Scale150X,
        Scale175X,
        Scale200X,
    }

    public enum AfterPlaybackSettingsOpt
    {
        DoNothing = 0,
        ClosePlayer,
        PlayNextFileInFolder,
        RepeatPlaylist
    }

    public enum AfterPlaybackSettingsAction
    {
        DoNothing = 0,
        GreyOutFile,
        RemoveFile
    }

    #endregion

    #region PlaylistSettings

    public class PlaylistSettings
    {
        public bool ShowPlaylistOnStartup { get; set; }
        public AfterPlaybackSettingsOpt AfterPlaybackOpt { get; set; }
        public AfterPlaybackSettingsAction AfterPlaybackAction { get; set; }
        public IconScale IconScale { get; set; }
        public bool BeginPlaybackOnStartup { get; set; }
        public bool RememberWindowSize { get; set; }
        public bool RememberWindowPosition { get; set; }
        public bool ShowToolTips { get; set; }
        public bool SnapWithPlayer { get; set; }
        public bool ScaleWithPlayer { get; set; }
        public bool StaySnapped { get; set; }
        public bool RememberPlaylist { get; set; }
        public bool StripDirectoryInFileName { get; set; }
        public Point WindowPosition { get; set; }
        public Size WindowSize { get; set; }
        public bool LockWindowSize { get; set; }
        public bool RememberColumns { get; set; }
        public List<string> Columns { get; set; }
        public List<string> RememberedFiles { get; set; }
        public List<string> RegexList { get; set; }
        public string Theme { get; set; }

        public PlaylistSettings()
        {
            int dpi = (int)GetDpi();
            switch (dpi)
            {
                case 96:
                    IconScale = IconScale.Scale100X;
                    break;
                case 120:
                    IconScale = IconScale.Scale150X;
                    break;
                case 144:
                case 192:
                case 240:
                    IconScale = IconScale.Scale200X;
                    break;
                default:
                    IconScale = IconScale.Scale100X;
                    break;
            }

            ShowPlaylistOnStartup = false;
            AfterPlaybackOpt = AfterPlaybackSettingsOpt.DoNothing;
            AfterPlaybackAction = AfterPlaybackSettingsAction.DoNothing;
            BeginPlaybackOnStartup = false;
            ShowToolTips = true;
            SnapWithPlayer = true;
            StaySnapped = false;
            RememberColumns = false;
            RememberWindowPosition = false;
            RememberWindowSize = false;
            LockWindowSize = false;
            RememberPlaylist = false;
            StripDirectoryInFileName = false;
            Columns = new List<string>();
            RememberedFiles = new List<string>();
            RegexList = new List<string>();
            Theme = "Default";
        }

        private float GetDpi()
        {
            var g = Graphics.FromHwnd(IntPtr.Zero);
            return g.DpiX;
        }
    }

    #endregion
}
