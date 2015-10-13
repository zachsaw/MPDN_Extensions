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
using System.Windows.Forms;
using Mpdn.Extensions.Framework.Config;
using Mpdn.Extensions.PlayerExtensions.Playlist;

namespace Mpdn.Extensions.PlayerExtensions
{
    public partial class PlaylistConfigDialog : PlaylistConfigBase
    {
        private static Form s_RegexForm;
        private int m_RegexCount;

        private int prevAfterPlaybackAction = -1;

        public PlaylistConfigDialog()
        {
            InitializeComponent();
            GetThemes();
            UpdateControls();

            Shown += PlaylistConfigDialog_Shown;
        }

        protected override void LoadSettings()
        {
            cb_showPlaylistOnStartup.Checked = Settings.ShowPlaylistOnStartup;
            cb_preload.Checked = Settings.PreloadNextFile;
            cb_afterPlaybackOpt.SelectedIndex = (int)Settings.AfterPlaybackOpt;
            cb_afterPlaybackAction.SelectedIndex = (int)Settings.AfterPlaybackAction;
            cb_iconScale.SelectedIndex = (int)Settings.IconScale;
            cb_onStartup.Checked = Settings.BeginPlaybackOnStartup;
            cb_showToolTips.Checked = Settings.ShowToolTips;
            cb_scaleWithPlayer.Checked = Settings.ScaleWithPlayer;
            cb_snapWithPlayer.Checked = Settings.SnapWithPlayer;
            cb_snapLocation.SelectedIndex = (int)Settings.SnapLocation;
            cb_staySnapped.Checked = Settings.StaySnapped;
            cb_rememberColumns.Checked = Settings.RememberColumns;
            cb_rememberWindowPosition.Checked = Settings.RememberWindowPosition;
            cb_rememberWindowSize.Checked = Settings.RememberWindowSize;
            cb_lockWindowSize.Checked = Settings.LockWindowSize;
            cb_rememberPlaylist.Checked = Settings.RememberPlaylist;
            cb_theme.Text = Settings.Theme;
        }

        protected override void SaveSettings()
        {
            Settings.ShowPlaylistOnStartup = cb_showPlaylistOnStartup.Checked;
            Settings.PreloadNextFile = cb_preload.Checked;
            Settings.AfterPlaybackOpt = (AfterPlaybackSettingsOpt)cb_afterPlaybackOpt.SelectedIndex;
            Settings.AfterPlaybackAction = (AfterPlaybackSettingsAction)cb_afterPlaybackAction.SelectedIndex;
            Settings.IconScale = (IconScale)cb_iconScale.SelectedIndex;
            Settings.BeginPlaybackOnStartup = cb_onStartup.Checked;
            Settings.ShowToolTips = cb_showToolTips.Checked;
            Settings.ScaleWithPlayer = cb_scaleWithPlayer.Checked;
            Settings.SnapWithPlayer = cb_snapWithPlayer.Checked;
            Settings.SnapLocation = (SnapLocation)cb_snapLocation.SelectedIndex;
            Settings.StaySnapped = cb_staySnapped.Checked;
            Settings.RememberColumns = cb_rememberColumns.Checked;
            Settings.RememberWindowPosition = cb_rememberWindowPosition.Checked;
            Settings.RememberWindowSize = cb_rememberWindowSize.Checked;
            Settings.LockWindowSize = cb_lockWindowSize.Checked;
            Settings.RememberPlaylist = cb_rememberPlaylist.Checked;
            Settings.Theme = cb_theme.Text;
        }

        private void GetThemes()
        {
            const string dir = PlaylistForm.PLAYLIST_ICONS_DIR;
            if (!Directory.Exists(dir)) return;

            var directories = Directory.EnumerateDirectories(Application.StartupPath + @"\" + dir);

            foreach (var d in directories)
            {
                cb_theme.Items.Add(new DirectoryInfo(d).Name);
            }
        }

        private void InitControls()
        {
            ValidateAfterPlaybackAction();
            ValidateAfterPlaybackOpt();
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
            var toolTip = new ToolTip();

            m_RegexCount = 0;

            s_RegexForm = new Form
            {
                Text = "Configure regex",
                Size = new Size(280, 270),
                ControlBox = false,
                FormBorderStyle = FormBorderStyle.FixedDialog,
                StartPosition = FormStartPosition.CenterScreen
            };

            var label = new Label
            {
                Text =
                    "Use regular expressions to filter out patterns or literal\nstrings you don't want to see in the Playlist.",
                Location = new Point(5, 5),
                AutoSize = true
            };

            var flowPanel = new FlowLayoutPanel
            {
                Size = new Size(274, 130),
                AutoSize = false,
                Location = new Point(0, 35),
                FlowDirection = FlowDirection.TopDown,
                AutoScroll = true,
                WrapContents = false
            };

            var btn_save = new Button
            {
                Text = "Save",
                Location = new Point(s_RegexForm.Width - 160, s_RegexForm.Height - 55)
            };

            var btn_close = new Button
            {
                Text = "Close",
                Location = new Point(btn_save.Location.X + btn_save.Size.Width + 2, btn_save.Location.Y)
            };

            var btn_clear = new Button
            {
                Text = "Clear",
                Location = new Point(btn_close.Location.X, btn_close.Location.Y - 25)
            };

            var btn_add = new Button
            {
                Text = "Add regex",
                Location = new Point(2, btn_save.Location.Y)
            };

            var cb_stripDirectory = new CheckBox
            {
                Text = "Strip directory in filename",
                AutoSize = true,
                Location = new Point(3, flowPanel.Location.Y + flowPanel.Height + 5)
            };

            toolTip.SetToolTip(cb_stripDirectory, "Strips the directory from the filename when there's more than one file in the directory");

            var link = new LinkLabel.Link
            {
                LinkData = "https://regex101.com/"
            };

            var linkLabel = new LinkLabel
            {
                Text = "Online Regex Building Tool",
                AutoSize = true,
                Links = {link},
                Location = new Point(2, cb_stripDirectory.Location.Y + cb_stripDirectory.Size.Height + 3)
            };

            if (Settings.RegexList != null && Settings.RegexList.Count > 0)
            {
                for (var i = 0; i < Settings.RegexList.Count; i++)
                {
                    m_RegexCount++;
                    flowPanel.Controls.Add(CreateRegexControls(i));
                }
            }
            else
            {
                for (var i = 0; i < 4; i++)
                {
                    m_RegexCount++;
                    flowPanel.Controls.Add(CreateRegexControls(i));
                }
            }

            if (Settings.StripDirectoryInFileName) cb_stripDirectory.Checked = true;

            btn_save.Click += btn_save_Click;
            btn_clear.Click += btn_clear_Click;
            btn_close.Click += btn_close_Click;
            btn_add.Click += btn_add_Click;
            linkLabel.LinkClicked += linkLabel_LinkClicked;

            s_RegexForm.Controls.Add(label);
            s_RegexForm.Controls.Add(flowPanel);
            s_RegexForm.Controls.Add(btn_save);
            s_RegexForm.Controls.Add(btn_close);
            s_RegexForm.Controls.Add(btn_clear);
            s_RegexForm.Controls.Add(btn_add);
            s_RegexForm.Controls.Add(cb_stripDirectory);
            s_RegexForm.Controls.Add(linkLabel);
            s_RegexForm.ShowDialog();
        }

        private Panel CreateRegexControls()
        {
            var panel = new Panel
            {
                Size = new Size(250, 24)
            };

            var label = new Label
            {
                Text = "Regex " + m_RegexCount + ":",
                Location = new Point(0, 3),
                Size = new Size(57, 30),
                AutoSize = false
            };

            var txtBox = new TextBox
            {
                Location = new Point(label.Location.X + label.Size.Width + 2, label.Location.Y - 3),
                Size = new Size(190, 30)
            };

            panel.Controls.AddRange(new Control[] {label, txtBox});
            return panel;
        }

        private Panel CreateRegexControls(int i)
        {
            var panel = new Panel
            {
                Size = new Size(250, 24)
            };

            var label = new Label
            {
                Text = "Regex " + m_RegexCount + ":",
                Location = new Point(0, 3),
                Size = new Size(57, 30),
                AutoSize = false
            };

            var txtBox = new TextBox
            {
                Location = new Point(label.Location.X + label.Size.Width + 2, label.Location.Y - 3),
                Size = new Size(190, 30)
            };

            if (Settings.RegexList != null && Settings.RegexList.Count > 0) if (i < Settings.RegexList.Count) txtBox.Text = Settings.RegexList[i];

            panel.Controls.AddRange(new Control[] {label, txtBox});
            return panel;
        }
        
        private void SaveRegex()
        {
            var regexList = new List<string>();

            foreach (var c in s_RegexForm.Controls)
            {
                var cb = c as CheckBox;
                if (cb == null) continue;

                Settings.StripDirectoryInFileName = cb.Checked;
            }

            foreach (var c in s_RegexForm.Controls)
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

        private void ClearRegex()
        {
            foreach (var c in s_RegexForm.Controls)
            {
                var cb = c as CheckBox;
                if (cb == null) continue;

                Settings.StripDirectoryInFileName = cb.Checked;
            }

            foreach (var c in s_RegexForm.Controls)
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
                        if (t != null) if (!string.IsNullOrEmpty(t.Text)) t.Text = "";
                    }
                }
            }

            PlaylistForm.UpdatePlaylistWithRegexFilter(new List<string>(), Settings.StripDirectoryInFileName);
        }

        private void ValidateAfterPlaybackAction()
        {
            bool isOnRepeat = ((AfterPlaybackSettingsOpt)cb_afterPlaybackOpt.SelectedIndex ==
                               AfterPlaybackSettingsOpt.RepeatPlaylist);
            bool isOnPlayNextFileInFolder = ((AfterPlaybackSettingsOpt)cb_afterPlaybackOpt.SelectedIndex ==
                                             AfterPlaybackSettingsOpt.PlayNextFileInFolder);

            int removeFileIdx = cb_afterPlaybackAction.FindString("Remove file");

            prevAfterPlaybackAction = cb_afterPlaybackAction.SelectedIndex;

            if (cb_afterPlaybackAction.SelectedIndex == removeFileIdx && (isOnRepeat || isOnPlayNextFileInFolder))
                cb_afterPlaybackAction.SelectedIndex = 0;
            else
                if (prevAfterPlaybackAction > -1) cb_afterPlaybackAction.SelectedIndex = prevAfterPlaybackAction;
        }

        private void ValidateAfterPlaybackOpt()
        {
            bool isOnRemove = ((AfterPlaybackSettingsAction)cb_afterPlaybackAction.SelectedIndex ==
                               AfterPlaybackSettingsAction.RemoveFile);


            int repeatPlaylistIdx = cb_afterPlaybackOpt.FindString("Repeat playlist");
            int playNextFileInFolderIdx = cb_afterPlaybackOpt.FindString("Play next file in folder");
            int idx = cb_afterPlaybackOpt.SelectedIndex;
            if ((idx == repeatPlaylistIdx || idx == playNextFileInFolderIdx) && isOnRemove)
                cb_afterPlaybackOpt.SelectedIndex = 0;
        }

        private void PlaylistConfigDialog_Shown(object sender, EventArgs e)
        {
            InitControls();
        }

        private void btn_save_Click(object sender, EventArgs e)
        {
            SaveRegex();
        }

        private void btn_close_Click(object sender, EventArgs e)
        {
            s_RegexForm.Close();
        }

        private void btn_clear_Click(object sender, EventArgs e)
        {
            var result = MessageBox.Show("Are you sure you want to clear all regex?", "Regex clearing", MessageBoxButtons.YesNo, MessageBoxIcon.Asterisk);
            if (result == DialogResult.Yes) ClearRegex();
        }

        private void btn_add_Click(object sender, EventArgs e)
        {
            m_RegexCount++;

            foreach (var c in s_RegexForm.Controls)
            {
                var flowPanel = c as FlowLayoutPanel;
                if (flowPanel == null) continue;

                var control = CreateRegexControls();
                flowPanel.Controls.Add(control);
                flowPanel.VerticalScroll.Value = flowPanel.VerticalScroll.Maximum;
                flowPanel.PerformLayout();
                control.Controls[1].Focus();
            }

            s_RegexForm.Invalidate();
        }

        private void linkLabel_LinkClicked(object sender, LinkLabelLinkClickedEventArgs e)
        {
            Process.Start(e.Link.LinkData as string);
        }

        private void cb_snapWithPlayer_CheckedChanged(object sender, EventArgs e)
        {
            UpdateControls();
        }

        private void btn_configRegex_Clicked(object sender, EventArgs e)
        {
            InitRegexForm();
        }

        private void cb_afterPlaybackOpt_DrawItem(object sender, DrawItemEventArgs e)
        {
            var comboBox = (ComboBox)sender;

            int repeatPlaylistIdx = cb_afterPlaybackOpt.FindString("Repeat playlist");
            int playNextFileInFolderIdx = cb_afterPlaybackOpt.FindString("Play next file in folder");

            bool isOnRemove = ((AfterPlaybackSettingsAction)cb_afterPlaybackAction.SelectedIndex == AfterPlaybackSettingsAction.RemoveFile);

            if ((e.Index == repeatPlaylistIdx || e.Index == playNextFileInFolderIdx) && isOnRemove)
            {
                e.Graphics.FillRectangle(SystemBrushes.Window, e.Bounds);
                e.Graphics.DrawString(comboBox.Items[e.Index].ToString(), comboBox.Font, SystemBrushes.GrayText, e.Bounds);
            }
            else
            {
                e.DrawBackground();

                if ((e.State & DrawItemState.Selected) == DrawItemState.Selected) e.Graphics.DrawString(comboBox.Items[e.Index].ToString(), comboBox.Font, SystemBrushes.Window, e.Bounds);
                else e.Graphics.DrawString(comboBox.Items[e.Index].ToString(), comboBox.Font, SystemBrushes.ControlText, e.Bounds);
                e.DrawFocusRectangle();
            }
        }

        private void cb_afterPlaybackAction_DrawItem(object sender, DrawItemEventArgs e)
        {
            var comboBox = (ComboBox)sender;

            int removeFileIdx = cb_afterPlaybackAction.FindString("Remove file");
            bool isOnRepeat = ((AfterPlaybackSettingsOpt)cb_afterPlaybackOpt.SelectedIndex == AfterPlaybackSettingsOpt.RepeatPlaylist);
            bool isOnPlayNextFileInFolder = ((AfterPlaybackSettingsOpt)cb_afterPlaybackOpt.SelectedIndex == AfterPlaybackSettingsOpt.PlayNextFileInFolder);

            if (e.Index == removeFileIdx && (isOnRepeat || isOnPlayNextFileInFolder))
            {
                e.Graphics.FillRectangle(SystemBrushes.Window, e.Bounds);
                e.Graphics.DrawString(comboBox.Items[e.Index].ToString(), comboBox.Font, SystemBrushes.GrayText, e.Bounds);
            }
            else
            {
                e.DrawBackground();

                if ((e.State & DrawItemState.Selected) == DrawItemState.Selected) e.Graphics.DrawString(comboBox.Items[e.Index].ToString(), comboBox.Font, SystemBrushes.Window, e.Bounds);
                else e.Graphics.DrawString(comboBox.Items[e.Index].ToString(), comboBox.Font, SystemBrushes.ControlText, e.Bounds);
                e.DrawFocusRectangle();
            }
        }

        private void cb_afterPlaybackOpt_SelectionChangeCommitted(object sender, EventArgs e)
        {
            ValidateAfterPlaybackOpt();
        }

        private void cb_afterPlaybackAction_SelectionChangeCommitted(object sender, EventArgs e)
        {
            ValidateAfterPlaybackAction();
        }
    }

    public class PlaylistConfigBase : ScriptConfigDialog<PlaylistSettings> {}
}
