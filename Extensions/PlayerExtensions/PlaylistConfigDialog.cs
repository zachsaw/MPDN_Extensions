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
using Mpdn.Extensions.Framework.Config;
using Mpdn.Extensions.PlayerExtensions.Playlist;

namespace Mpdn.Extensions.PlayerExtensions
{
    public partial class PlaylistConfigDialog : PlaylistConfigBase
    {
        public PlaylistConfigDialog()
        {
            InitializeComponent();
            updateControls();
        }

        protected override void LoadSettings()
        {
            cb_showPlaylistOnStartup.Checked = Settings.ShowPlaylistOnStartup;
            cb_afterPlaybackOpt.SelectedIndex = (int)Settings.AfterPlaybackOpt;
            cb_onStartup.Checked = Settings.BeginPlaybackOnStartup;
            cb_whenFileIsAdded.Checked = Settings.BeginPlaybackWhenFileIsAdded;
            cb_whenPlaylistFileIsOpened.Checked = Settings.BeginPlaybackWhenPlaylistFileIsOpened;
            cb_scaleWithPlayer.Checked = Settings.ScaleWithPlayer;
            cb_snapWithPlayer.Checked = Settings.SnapWithPlayer;
            cb_staySnapped.Checked = Settings.StaySnapped;
            cb_rememberColumns.Checked = Settings.RememberColumns;
            cb_rememberWindowPosition.Checked = Settings.RememberWindowPosition;
            cb_rememberWindowSize.Checked = Settings.RememberWindowSize;
            cb_lockWindowSize.Checked = Settings.LockWindowSize;
            cb_rememberPlaylist.Checked = Settings.RememberPlaylist;
            cb_addToPlaylistOnFileOpen.Checked = Settings.AddToPlaylistOnFileOpen;
        }

        protected override void SaveSettings()
        {
            Settings.ShowPlaylistOnStartup = cb_showPlaylistOnStartup.Checked;
            Settings.AfterPlaybackOpt = (AfterPlaybackSettingsOpt)cb_afterPlaybackOpt.SelectedIndex;
            Settings.BeginPlaybackOnStartup = cb_onStartup.Checked;
            Settings.BeginPlaybackWhenFileIsAdded = cb_whenFileIsAdded.Checked;
            Settings.BeginPlaybackWhenPlaylistFileIsOpened = cb_whenPlaylistFileIsOpened.Checked;
            Settings.ScaleWithPlayer = cb_scaleWithPlayer.Checked;
            Settings.SnapWithPlayer = cb_snapWithPlayer.Checked;
            Settings.StaySnapped = cb_staySnapped.Checked;
            Settings.RememberColumns = cb_rememberColumns.Checked;
            Settings.RememberWindowPosition = cb_rememberWindowPosition.Checked;
            Settings.RememberWindowSize = cb_rememberWindowSize.Checked;
            Settings.LockWindowSize = cb_lockWindowSize.Checked;
            Settings.RememberPlaylist = cb_rememberPlaylist.Checked;
            Settings.AddToPlaylistOnFileOpen = cb_addToPlaylistOnFileOpen.Checked;
        }

        private void updateControls()
        {
            if (cb_snapWithPlayer.Checked) cb_staySnapped.Enabled = true;
            else { cb_staySnapped.Checked = false; cb_staySnapped.Enabled = false; }
        }

        private void cb_snapWithPlayer_CheckedChanged(object sender, System.EventArgs e)
        {
            updateControls();
        }
    }

    public class PlaylistConfigBase : ScriptConfigDialog<PlaylistSettings>
    {
        
    }
}
