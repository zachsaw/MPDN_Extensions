using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using Mpdn.Config;

namespace Mpdn.PlayerExtensions.Playlist
{
    public partial class PlaylistConfigDialog : PlaylistConfigBase
    {
        public PlaylistConfigDialog()
        {
            InitializeComponent();
        }

        protected override void LoadSettings()
        {
            cb_rememberWindowBounds.Checked = Settings.RememberWindowBounds;
            cb_rememberLastPlayedFile.Checked = Settings.RememberLastPlayedFile;
            cb_addFileToPlaylistOnOpen.Checked = Settings.AddFileToPlaylistOnOpen;
        }

        protected override void SaveSettings()
        {
            Settings.RememberWindowBounds = cb_rememberWindowBounds.Checked;
            Settings.RememberLastPlayedFile = cb_rememberLastPlayedFile.Checked;
            Settings.AddFileToPlaylistOnOpen = cb_addFileToPlaylistOnOpen.Checked;
        }
    }

    public class PlaylistConfigBase : ScriptConfigDialog<PlaylistSettings>
    {
        
    }
}
