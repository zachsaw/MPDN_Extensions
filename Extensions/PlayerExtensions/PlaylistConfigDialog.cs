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
using System.Windows.Forms;
using Mpdn.Extensions.Framework.Config;
using Mpdn.Extensions.PlayerExtensions.Playlist;

namespace Mpdn.Extensions.PlayerExtensions
{
    public partial class PlaylistConfigDialog : PlaylistConfigBase
    {
        private static Form regexForm;

        public PlaylistConfigDialog()
        {
            InitializeComponent();
            UpdateControls();
        }

        protected override void LoadSettings()
        {
            cb_showPlaylistOnStartup.Checked = Settings.ShowPlaylistOnStartup;
            cb_afterPlaybackOpt.SelectedIndex = (int)Settings.AfterPlaybackOpt;
            cb_onStartup.Checked = Settings.BeginPlaybackOnStartup;
            cb_scaleWithPlayer.Checked = Settings.ScaleWithPlayer;
            cb_snapWithPlayer.Checked = Settings.SnapWithPlayer;
            cb_staySnapped.Checked = Settings.StaySnapped;
            cb_rememberColumns.Checked = Settings.RememberColumns;
            cb_rememberWindowPosition.Checked = Settings.RememberWindowPosition;
            cb_rememberWindowSize.Checked = Settings.RememberWindowSize;
            cb_lockWindowSize.Checked = Settings.LockWindowSize;
            cb_rememberPlaylist.Checked = Settings.RememberPlaylist;
        }

        protected override void SaveSettings()
        {
            Settings.ShowPlaylistOnStartup = cb_showPlaylistOnStartup.Checked;
            Settings.AfterPlaybackOpt = (AfterPlaybackSettingsOpt)cb_afterPlaybackOpt.SelectedIndex;
            Settings.BeginPlaybackOnStartup = cb_onStartup.Checked;
            Settings.ScaleWithPlayer = cb_scaleWithPlayer.Checked;
            Settings.SnapWithPlayer = cb_snapWithPlayer.Checked;
            Settings.StaySnapped = cb_staySnapped.Checked;
            Settings.RememberColumns = cb_rememberColumns.Checked;
            Settings.RememberWindowPosition = cb_rememberWindowPosition.Checked;
            Settings.RememberWindowSize = cb_rememberWindowSize.Checked;
            Settings.LockWindowSize = cb_lockWindowSize.Checked;
            Settings.RememberPlaylist = cb_rememberPlaylist.Checked;
        }

        private void UpdateControls()
        {
            if (cb_snapWithPlayer.Checked)
            {
                cb_staySnapped.Enabled = true;
                cb_scaleWithPlayer.Enabled = true;
                cb_rememberWindowPosition.Checked = false;
                cb_rememberWindowPosition.Enabled = false;
            }

            else
            {
                cb_staySnapped.Checked = false;
                cb_staySnapped.Enabled = false;
                cb_scaleWithPlayer.Checked = false;
                cb_scaleWithPlayer.Enabled = false;
                cb_rememberWindowPosition.Enabled = true;
            }
        }

        private void InitRegexForm()
        {
            regexForm = new Form
            {
                Text = "Configure regex",
                Size = new Size(280, 230),
                ControlBox = false,
                FormBorderStyle = FormBorderStyle.FixedDialog,
                StartPosition = FormStartPosition.CenterScreen
            };

            var flowPanel = new FlowLayoutPanel
            {
                Size = new Size(274, 130),
                AutoSize = false,
                Location = new Point(0, 10),
                FlowDirection = FlowDirection.TopDown,
                AutoScroll = true,
                WrapContents = false
            };

            var btn_apply = new Button
            {
                Name = "btn_apply",
                Text = "Apply",
                Location = new Point(regexForm.Width - 160, regexForm.Height - 55)
            };

            var btn_close = new Button
            {
                Name = "btn_close",
                Text = "Close",
                Location = new Point(btn_apply.Location.X + btn_apply.Size.Width + 2, btn_apply.Location.Y)
            };

            var btn_add = new Button()
            {
                Name = "btn_add",
                Text = "Add regex",
                Location = new Point(regexForm.Left + 3, btn_apply.Location.Y)
            };

            var cb_stripDirectory = new CheckBox
            {
                Name = "cb_stripDirectory",
                Text = "Strip directory in filename",
                AutoSize = true,
                Location = new Point(5, flowPanel.Height + 15)
            };

            if (Settings.RegexList != null && Settings.RegexList.Count > 0)
            {
                for (var i = 0; i < Settings.RegexList.Count; i++)
                {
                    flowPanel.Controls.Add(CreateRegexControls(i));
                }
            }
            else
            {
                for (var i = 0; i < 4; i++)
                {
                    flowPanel.Controls.Add(CreateRegexControls(i));
                }
            }

            if (Settings.StripDirectoryInFileName) cb_stripDirectory.Checked = true;

            btn_apply.Click += btn_apply_Click;
            btn_close.Click += btn_close_Click;
            btn_add.Click += btn_add_Click;

            regexForm.Controls.Add(flowPanel);
            regexForm.Controls.Add(btn_apply);
            regexForm.Controls.Add(btn_close);
            regexForm.Controls.Add(btn_add);
            regexForm.Controls.Add(cb_stripDirectory);
            regexForm.ShowDialog();
        }

        private Panel CreateRegexControls()
        {
            var panel = new Panel
            {
                Size = new Size(245, 24)
            };

            var label = new Label
            {
                Text = "Regex:",
                Location = new Point(3, 3),
                Size = new Size(42, 30),
                AutoSize = false
            };

            var txtBox = new TextBox
            {
                Location = new Point(label.Location.X + label.Size.Width + 10, label.Location.Y - 3),
                Size = new Size(190, 30)
            };

            panel.Controls.AddRange(new Control[] { label, txtBox });
            return panel;
        }

        private Panel CreateRegexControls(int i)
        {
            var panel = new Panel
            {
                Size = new Size(245, 24)
            };

            var label = new Label
            {
                Text = "Regex:",
                Location = new Point(3, 3),
                Size = new Size(42, 30),
                AutoSize = false
            };

            var txtBox = new TextBox
            {
                Name = "tb_regex" + i,
                Location = new Point(label.Location.X + label.Size.Width + 10, label.Location.Y - 3),
                Size = new Size(190, 30)
            };

            if (Settings.RegexList != null && Settings.RegexList.Count > 0)
            {
                if (i < Settings.RegexList.Count) txtBox.Text = Settings.RegexList[i];
            }

            panel.Controls.AddRange(new Control[] {label, txtBox});
            return panel;
        }

        private void btn_apply_Click(object sender, EventArgs e)
        {
            var regexList = new List<string>();

            foreach (var c in regexForm.Controls)
            {
                var cb = c as CheckBox;
                if (cb == null) continue;

                Settings.StripDirectoryInFileName = cb.Checked;
            }

            foreach (var c in regexForm.Controls)
            {
                var flowPanel = c as FlowLayoutPanel;
                if (flowPanel == null) continue;

                foreach (var c1 in flowPanel.Controls)
                {
                    var panel = c1 as Panel;
                    if (panel == null) continue;

                    foreach (var c2 in panel.Controls)
                    {
                        var t = c2 as TextBox;
                        if (t != null) if (!string.IsNullOrEmpty(t.Text)) regexList.Add(t.Text);
                    }
                }
            }

            PlaylistForm.UpdatePlaylistWithRegexFilter(regexList, Settings.StripDirectoryInFileName);
        }
        private void btn_close_Click(object sender, EventArgs e)
        {
            regexForm.Close();
        }

        private void btn_add_Click(object sender, EventArgs e)
        {
            foreach (var c in regexForm.Controls)
            {
                var flowPanel = c as FlowLayoutPanel;
                if (flowPanel == null) continue;

                flowPanel.Controls.Add(CreateRegexControls());
            }

            regexForm.Invalidate();
        }

        private void cb_snapWithPlayer_CheckedChanged(object sender, EventArgs e)
        {
            UpdateControls();
        }

        private void btn_configRegex_Clicked(object sender, EventArgs e)
        {
            InitRegexForm();
        }
    }

    public class PlaylistConfigBase : ScriptConfigDialog<PlaylistSettings> {}
}
