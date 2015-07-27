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
﻿using System.Windows.Forms;

namespace Mpdn.Extensions.PlayerExtensions
{
    partial class PlaylistConfigDialog
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.cb_showPlaylistOnStartup = new System.Windows.Forms.CheckBox();
            this.cb_rememberPlaylist = new System.Windows.Forms.CheckBox();
            this.cb_rememberWindowPosition = new System.Windows.Forms.CheckBox();
            this.btn_save = new System.Windows.Forms.Button();
            this.btn_cancel = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.groupBox4 = new System.Windows.Forms.GroupBox();
            this.cb_afterPlaybackAction = new System.Windows.Forms.ComboBox();
            this.cb_afterPlaybackOpt = new System.Windows.Forms.ComboBox();
            this.groupBox3 = new System.Windows.Forms.GroupBox();
            this.cb_onStartup = new System.Windows.Forms.CheckBox();
            this.panel1 = new System.Windows.Forms.Panel();
            this.label3 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.groupBox5 = new System.Windows.Forms.GroupBox();
            this.groupBox7 = new System.Windows.Forms.GroupBox();
            this.cb_iconScale = new System.Windows.Forms.ComboBox();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.cb_theme = new System.Windows.Forms.ComboBox();
            this.cb_showToolTips = new System.Windows.Forms.CheckBox();
            this.cb_staySnapped = new System.Windows.Forms.CheckBox();
            this.cb_scaleWithPlayer = new System.Windows.Forms.CheckBox();
            this.cb_lockWindowSize = new System.Windows.Forms.CheckBox();
            this.cb_snapWithPlayer = new System.Windows.Forms.CheckBox();
            this.groupBox6 = new System.Windows.Forms.GroupBox();
            this.cb_rememberColumns = new System.Windows.Forms.CheckBox();
            this.cb_rememberWindowSize = new System.Windows.Forms.CheckBox();
            this.btn_configRegex = new System.Windows.Forms.Button();
            this.groupBox2.SuspendLayout();
            this.groupBox4.SuspendLayout();
            this.groupBox3.SuspendLayout();
            this.panel1.SuspendLayout();
            this.groupBox5.SuspendLayout();
            this.groupBox7.SuspendLayout();
            this.groupBox1.SuspendLayout();
            this.groupBox6.SuspendLayout();
            this.SuspendLayout();
            // 
            // cb_showPlaylistOnStartup
            // 
            this.cb_showPlaylistOnStartup.AutoSize = true;
            this.cb_showPlaylistOnStartup.Location = new System.Drawing.Point(9, 70);
            this.cb_showPlaylistOnStartup.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.cb_showPlaylistOnStartup.Name = "cb_showPlaylistOnStartup";
            this.cb_showPlaylistOnStartup.Size = new System.Drawing.Size(179, 21);
            this.cb_showPlaylistOnStartup.TabIndex = 1;
            this.cb_showPlaylistOnStartup.Text = "Show playlist on startup";
            this.cb_showPlaylistOnStartup.UseVisualStyleBackColor = true;
            // 
            // cb_rememberPlaylist
            // 
            this.cb_rememberPlaylist.AutoSize = true;
            this.cb_rememberPlaylist.Location = new System.Drawing.Point(9, 43);
            this.cb_rememberPlaylist.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.cb_rememberPlaylist.Name = "cb_rememberPlaylist";
            this.cb_rememberPlaylist.Size = new System.Drawing.Size(146, 21);
            this.cb_rememberPlaylist.TabIndex = 5;
            this.cb_rememberPlaylist.Text = "Remember playlist";
            this.cb_rememberPlaylist.UseVisualStyleBackColor = true;
            // 
            // cb_rememberWindowPosition
            // 
            this.cb_rememberWindowPosition.AutoSize = true;
            this.cb_rememberWindowPosition.Location = new System.Drawing.Point(9, 66);
            this.cb_rememberWindowPosition.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.cb_rememberWindowPosition.Name = "cb_rememberWindowPosition";
            this.cb_rememberWindowPosition.Size = new System.Drawing.Size(201, 21);
            this.cb_rememberWindowPosition.TabIndex = 6;
            this.cb_rememberWindowPosition.Text = "Remember window position";
            this.cb_rememberWindowPosition.UseVisualStyleBackColor = true;
            // 
            // btn_save
            // 
            this.btn_save.BackColor = System.Drawing.SystemColors.ButtonFace;
            this.btn_save.DialogResult = System.Windows.Forms.DialogResult.OK;
            this.btn_save.Location = new System.Drawing.Point(397, 377);
            this.btn_save.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.btn_save.Name = "btn_save";
            this.btn_save.Size = new System.Drawing.Size(100, 28);
            this.btn_save.TabIndex = 15;
            this.btn_save.Text = "Save";
            this.btn_save.UseVisualStyleBackColor = false;
            // 
            // btn_cancel
            // 
            this.btn_cancel.BackColor = System.Drawing.SystemColors.ButtonFace;
            this.btn_cancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
            this.btn_cancel.Location = new System.Drawing.Point(500, 377);
            this.btn_cancel.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.btn_cancel.Name = "btn_cancel";
            this.btn_cancel.Size = new System.Drawing.Size(100, 28);
            this.btn_cancel.TabIndex = 16;
            this.btn_cancel.Text = "Cancel";
            this.btn_cancel.UseVisualStyleBackColor = false;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label1.ForeColor = System.Drawing.SystemColors.ControlText;
            this.label1.Location = new System.Drawing.Point(13, 4);
            this.label1.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(81, 25);
            this.label1.TabIndex = 4;
            this.label1.Text = "Playlist";
            // 
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this.groupBox4);
            this.groupBox2.Controls.Add(this.groupBox3);
            this.groupBox2.Location = new System.Drawing.Point(301, 63);
            this.groupBox2.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Padding = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.groupBox2.Size = new System.Drawing.Size(299, 193);
            this.groupBox2.TabIndex = 8;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Playback";
            // 
            // groupBox4
            // 
            this.groupBox4.Controls.Add(this.cb_afterPlaybackAction);
            this.groupBox4.Controls.Add(this.cb_afterPlaybackOpt);
            this.groupBox4.Location = new System.Drawing.Point(8, 92);
            this.groupBox4.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.groupBox4.Name = "groupBox4";
            this.groupBox4.Padding = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.groupBox4.Size = new System.Drawing.Size(283, 91);
            this.groupBox4.TabIndex = 13;
            this.groupBox4.TabStop = false;
            this.groupBox4.Text = "After playback";
            // 
            // cb_afterPlaybackAction
            // 
            this.cb_afterPlaybackAction.DrawMode = System.Windows.Forms.DrawMode.OwnerDrawFixed;
            this.cb_afterPlaybackAction.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cb_afterPlaybackAction.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.cb_afterPlaybackAction.FormattingEnabled = true;
            this.cb_afterPlaybackAction.Items.AddRange(new object[] {
            "Do nothing",
            "Grey out file",
            "Remove file"});
            this.cb_afterPlaybackAction.Location = new System.Drawing.Point(8, 57);
            this.cb_afterPlaybackAction.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.cb_afterPlaybackAction.Name = "cb_afterPlaybackAction";
            this.cb_afterPlaybackAction.Size = new System.Drawing.Size(265, 23);
            this.cb_afterPlaybackAction.TabIndex = 15;
            this.cb_afterPlaybackAction.DrawItem += new System.Windows.Forms.DrawItemEventHandler(this.cb_afterPlaybackAction_DrawItem);
            this.cb_afterPlaybackAction.SelectionChangeCommitted += new System.EventHandler(this.cb_afterPlaybackAction_SelectionChangeCommitted);
            this.cb_afterPlaybackAction.Enter += new System.EventHandler(this.cb_afterPlaybackAction_Enter);
            // 
            // cb_afterPlaybackOpt
            // 
            this.cb_afterPlaybackOpt.DrawMode = System.Windows.Forms.DrawMode.OwnerDrawFixed;
            this.cb_afterPlaybackOpt.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cb_afterPlaybackOpt.FormattingEnabled = true;
            this.cb_afterPlaybackOpt.Items.AddRange(new object[] {
            "Do nothing",
            "Close player",
            "Play next file in folder",
            "Repeat playlist"});
            this.cb_afterPlaybackOpt.Location = new System.Drawing.Point(8, 21);
            this.cb_afterPlaybackOpt.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.cb_afterPlaybackOpt.Name = "cb_afterPlaybackOpt";
            this.cb_afterPlaybackOpt.Size = new System.Drawing.Size(265, 23);
            this.cb_afterPlaybackOpt.TabIndex = 14;
            this.cb_afterPlaybackOpt.DrawItem += new System.Windows.Forms.DrawItemEventHandler(this.cb_afterPlaybackOpt_DrawItem);
            this.cb_afterPlaybackOpt.SelectionChangeCommitted += new System.EventHandler(this.cb_afterPlaybackOpt_SelectionChangeCommitted);
            this.cb_afterPlaybackOpt.Enter += new System.EventHandler(this.cb_afterPlaybackOpt_Enter);
            // 
            // groupBox3
            // 
            this.groupBox3.Controls.Add(this.cb_onStartup);
            this.groupBox3.Location = new System.Drawing.Point(8, 22);
            this.groupBox3.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.groupBox3.Name = "groupBox3";
            this.groupBox3.Padding = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.groupBox3.Size = new System.Drawing.Size(283, 53);
            this.groupBox3.TabIndex = 9;
            this.groupBox3.TabStop = false;
            this.groupBox3.Text = "Begin playback";
            // 
            // cb_onStartup
            // 
            this.cb_onStartup.AutoSize = true;
            this.cb_onStartup.Location = new System.Drawing.Point(8, 23);
            this.cb_onStartup.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.cb_onStartup.Name = "cb_onStartup";
            this.cb_onStartup.Size = new System.Drawing.Size(97, 21);
            this.cb_onStartup.TabIndex = 10;
            this.cb_onStartup.Text = "On startup";
            this.cb_onStartup.UseVisualStyleBackColor = true;
            // 
            // panel1
            // 
            this.panel1.BackColor = System.Drawing.SystemColors.InactiveBorder;
            this.panel1.BorderStyle = System.Windows.Forms.BorderStyle.Fixed3D;
            this.panel1.Controls.Add(this.label3);
            this.panel1.Controls.Add(this.label1);
            this.panel1.Controls.Add(this.label2);
            this.panel1.Location = new System.Drawing.Point(-12, -2);
            this.panel1.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(632, 57);
            this.panel1.TabIndex = 7;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label3.ForeColor = System.Drawing.SystemColors.ActiveCaption;
            this.label3.Location = new System.Drawing.Point(403, 31);
            this.label3.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(214, 17);
            this.label3.TabIndex = 6;
            this.label3.Text = "Thanks to Zach, ryrynz, mrcorbo";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.ForeColor = System.Drawing.SystemColors.ControlText;
            this.label2.Location = new System.Drawing.Point(15, 31);
            this.label2.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(290, 17);
            this.label2.TabIndex = 5;
            this.label2.Text = "Playlist with advanced capabilities by Garteal";
            // 
            // groupBox5
            // 
            this.groupBox5.Controls.Add(this.groupBox7);
            this.groupBox5.Controls.Add(this.groupBox1);
            this.groupBox5.Controls.Add(this.cb_showToolTips);
            this.groupBox5.Controls.Add(this.cb_staySnapped);
            this.groupBox5.Controls.Add(this.cb_scaleWithPlayer);
            this.groupBox5.Controls.Add(this.cb_lockWindowSize);
            this.groupBox5.Controls.Add(this.cb_snapWithPlayer);
            this.groupBox5.Controls.Add(this.cb_showPlaylistOnStartup);
            this.groupBox5.Location = new System.Drawing.Point(9, 188);
            this.groupBox5.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.groupBox5.Name = "groupBox5";
            this.groupBox5.Padding = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.groupBox5.Size = new System.Drawing.Size(273, 215);
            this.groupBox5.TabIndex = 0;
            this.groupBox5.TabStop = false;
            this.groupBox5.Text = "Interface";
            // 
            // groupBox7
            // 
            this.groupBox7.Controls.Add(this.cb_iconScale);
            this.groupBox7.Location = new System.Drawing.Point(9, 145);
            this.groupBox7.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.groupBox7.Name = "groupBox7";
            this.groupBox7.Padding = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.groupBox7.Size = new System.Drawing.Size(100, 60);
            this.groupBox7.TabIndex = 19;
            this.groupBox7.TabStop = false;
            this.groupBox7.Text = "Icon scale";
            // 
            // cb_iconScale
            // 
            this.cb_iconScale.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cb_iconScale.FormattingEnabled = true;
            this.cb_iconScale.Items.AddRange(new object[] {
            "100%",
            "125%",
            "150%",
            "175%",
            "200%"});
            this.cb_iconScale.Location = new System.Drawing.Point(8, 22);
            this.cb_iconScale.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.cb_iconScale.Name = "cb_iconScale";
            this.cb_iconScale.Size = new System.Drawing.Size(83, 24);
            this.cb_iconScale.TabIndex = 18;
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.cb_theme);
            this.groupBox1.Location = new System.Drawing.Point(116, 145);
            this.groupBox1.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Padding = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.groupBox1.Size = new System.Drawing.Size(149, 60);
            this.groupBox1.TabIndex = 18;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Theme";
            // 
            // cb_theme
            // 
            this.cb_theme.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cb_theme.FormattingEnabled = true;
            this.cb_theme.Location = new System.Drawing.Point(8, 22);
            this.cb_theme.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.cb_theme.Name = "cb_theme";
            this.cb_theme.Size = new System.Drawing.Size(133, 24);
            this.cb_theme.TabIndex = 18;
            // 
            // cb_showToolTips
            // 
            this.cb_showToolTips.AutoSize = true;
            this.cb_showToolTips.Checked = true;
            this.cb_showToolTips.CheckState = System.Windows.Forms.CheckState.Checked;
            this.cb_showToolTips.Location = new System.Drawing.Point(9, 94);
            this.cb_showToolTips.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.cb_showToolTips.Name = "cb_showToolTips";
            this.cb_showToolTips.Size = new System.Drawing.Size(113, 21);
            this.cb_showToolTips.TabIndex = 6;
            this.cb_showToolTips.Text = "Show tooltips";
            this.cb_showToolTips.UseVisualStyleBackColor = true;
            // 
            // cb_staySnapped
            // 
            this.cb_staySnapped.AutoSize = true;
            this.cb_staySnapped.Location = new System.Drawing.Point(151, 117);
            this.cb_staySnapped.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.cb_staySnapped.Name = "cb_staySnapped";
            this.cb_staySnapped.Size = new System.Drawing.Size(117, 21);
            this.cb_staySnapped.TabIndex = 5;
            this.cb_staySnapped.Text = "Stay snapped";
            this.cb_staySnapped.UseVisualStyleBackColor = true;
            // 
            // cb_scaleWithPlayer
            // 
            this.cb_scaleWithPlayer.AutoSize = true;
            this.cb_scaleWithPlayer.Location = new System.Drawing.Point(9, 47);
            this.cb_scaleWithPlayer.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.cb_scaleWithPlayer.Name = "cb_scaleWithPlayer";
            this.cb_scaleWithPlayer.Size = new System.Drawing.Size(136, 21);
            this.cb_scaleWithPlayer.TabIndex = 4;
            this.cb_scaleWithPlayer.Text = "Scale with player";
            this.cb_scaleWithPlayer.UseVisualStyleBackColor = true;
            // 
            // cb_lockWindowSize
            // 
            this.cb_lockWindowSize.AutoSize = true;
            this.cb_lockWindowSize.Location = new System.Drawing.Point(9, 23);
            this.cb_lockWindowSize.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.cb_lockWindowSize.Name = "cb_lockWindowSize";
            this.cb_lockWindowSize.Size = new System.Drawing.Size(138, 21);
            this.cb_lockWindowSize.TabIndex = 3;
            this.cb_lockWindowSize.Text = "Lock window size";
            this.cb_lockWindowSize.UseVisualStyleBackColor = true;
            // 
            // cb_snapWithPlayer
            // 
            this.cb_snapWithPlayer.AutoSize = true;
            this.cb_snapWithPlayer.Checked = true;
            this.cb_snapWithPlayer.CheckState = System.Windows.Forms.CheckState.Checked;
            this.cb_snapWithPlayer.Location = new System.Drawing.Point(9, 117);
            this.cb_snapWithPlayer.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.cb_snapWithPlayer.Name = "cb_snapWithPlayer";
            this.cb_snapWithPlayer.Size = new System.Drawing.Size(134, 21);
            this.cb_snapWithPlayer.TabIndex = 2;
            this.cb_snapWithPlayer.Text = "Snap with player";
            this.cb_snapWithPlayer.UseVisualStyleBackColor = true;
            this.cb_snapWithPlayer.CheckedChanged += new System.EventHandler(this.cb_snapWithPlayer_CheckedChanged);
            // 
            // groupBox6
            // 
            this.groupBox6.Controls.Add(this.cb_rememberColumns);
            this.groupBox6.Controls.Add(this.cb_rememberWindowSize);
            this.groupBox6.Controls.Add(this.cb_rememberPlaylist);
            this.groupBox6.Controls.Add(this.cb_rememberWindowPosition);
            this.groupBox6.Location = new System.Drawing.Point(9, 63);
            this.groupBox6.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.groupBox6.Name = "groupBox6";
            this.groupBox6.Padding = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.groupBox6.Size = new System.Drawing.Size(273, 116);
            this.groupBox6.TabIndex = 4;
            this.groupBox6.TabStop = false;
            this.groupBox6.Text = "History";
            // 
            // cb_rememberColumns
            // 
            this.cb_rememberColumns.AutoSize = true;
            this.cb_rememberColumns.Location = new System.Drawing.Point(9, 20);
            this.cb_rememberColumns.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.cb_rememberColumns.Name = "cb_rememberColumns";
            this.cb_rememberColumns.Size = new System.Drawing.Size(155, 21);
            this.cb_rememberColumns.TabIndex = 8;
            this.cb_rememberColumns.Text = "Remember columns";
            this.cb_rememberColumns.UseVisualStyleBackColor = true;
            // 
            // cb_rememberWindowSize
            // 
            this.cb_rememberWindowSize.AutoSize = true;
            this.cb_rememberWindowSize.Location = new System.Drawing.Point(9, 89);
            this.cb_rememberWindowSize.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.cb_rememberWindowSize.Name = "cb_rememberWindowSize";
            this.cb_rememberWindowSize.Size = new System.Drawing.Size(177, 21);
            this.cb_rememberWindowSize.TabIndex = 7;
            this.cb_rememberWindowSize.Text = "Remember window size";
            this.cb_rememberWindowSize.UseVisualStyleBackColor = true;
            // 
            // btn_configRegex
            // 
            this.btn_configRegex.BackColor = System.Drawing.SystemColors.ButtonFace;
            this.btn_configRegex.Location = new System.Drawing.Point(295, 377);
            this.btn_configRegex.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.btn_configRegex.Name = "btn_configRegex";
            this.btn_configRegex.Size = new System.Drawing.Size(100, 28);
            this.btn_configRegex.TabIndex = 17;
            this.btn_configRegex.Text = "Config regex";
            this.btn_configRegex.UseVisualStyleBackColor = false;
            this.btn_configRegex.Click += new System.EventHandler(this.btn_configRegex_Clicked);
            // 
            // PlaylistConfigDialog
            // 
            this.AcceptButton = this.btn_save;
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.SystemColors.Window;
            this.ClientSize = new System.Drawing.Size(608, 415);
            this.Controls.Add(this.btn_configRegex);
            this.Controls.Add(this.groupBox6);
            this.Controls.Add(this.groupBox5);
            this.Controls.Add(this.panel1);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.btn_cancel);
            this.Controls.Add(this.btn_save);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "PlaylistConfigDialog";
            this.ShowIcon = false;
            this.ShowInTaskbar = false;
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
            this.Text = "Playlist Configuration";
            this.groupBox2.ResumeLayout(false);
            this.groupBox4.ResumeLayout(false);
            this.groupBox3.ResumeLayout(false);
            this.groupBox3.PerformLayout();
            this.panel1.ResumeLayout(false);
            this.panel1.PerformLayout();
            this.groupBox5.ResumeLayout(false);
            this.groupBox5.PerformLayout();
            this.groupBox7.ResumeLayout(false);
            this.groupBox1.ResumeLayout(false);
            this.groupBox6.ResumeLayout(false);
            this.groupBox6.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.CheckBox cb_rememberWindowPosition;
        private System.Windows.Forms.CheckBox cb_rememberPlaylist;
        private System.Windows.Forms.Button btn_save;
        private System.Windows.Forms.Button btn_cancel;
        private System.Windows.Forms.CheckBox cb_showPlaylistOnStartup;
        private System.Windows.Forms.Label label1;
        private GroupBox groupBox2;
        private Panel panel1;
        private Label label3;
        private Label label2;
        private GroupBox groupBox4;
        private ComboBox cb_afterPlaybackOpt;
        private GroupBox groupBox3;
        private GroupBox groupBox5;
        private GroupBox groupBox6;
        private CheckBox cb_onStartup;
        private CheckBox cb_rememberWindowSize;
        private CheckBox cb_snapWithPlayer;
        private CheckBox cb_rememberColumns;
        private CheckBox cb_lockWindowSize;
        private CheckBox cb_scaleWithPlayer;
        private CheckBox cb_staySnapped;
        private Button btn_configRegex;
        private ComboBox cb_afterPlaybackAction;
        private CheckBox cb_showToolTips;
        private GroupBox groupBox1;
        private ComboBox cb_theme;
        private GroupBox groupBox7;
        private ComboBox cb_iconScale;
    }
}
