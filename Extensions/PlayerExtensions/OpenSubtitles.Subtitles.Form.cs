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
using System.Linq;
using System.Windows.Forms;
using Mpdn.Extensions.Framework.Controls;

namespace Mpdn.Extensions.PlayerExtensions
{
    public partial class OpenSubtitlesForm : Form
    {
        private Subtitle m_SelectedSub;
        private List<Subtitle> m_SubtitleList; 

        public OpenSubtitlesForm()
        {
            InitializeComponent();
            dataGridView1.AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.DisplayedCells;
            CancelButton = btnCancel;
            Icon = PlayerControl.ApplicationIcon;
            FormClosed += OpenSubtitlesForm_FormClosed;
            comboBoxChangeLang.KeyDown += OpenSubtitlesLanguageHandler.ComboBoxPrefLanguageKeyDown;
            comboBoxChangeLang.KeyPress += OpenSubtitlesLanguageHandler.ComboBoxPrefLanguageKeyPress;
        }

        private void OpenSubtitlesForm_FormClosed(object sender, FormClosedEventArgs e)
        {
            m_SelectedSub = null;
            subtitleBindingSource.DataSource = typeof (Subtitle);
        }

        public void SetSubtitles(List<Subtitle> subtitles, String prefLang)
        {
            m_SubtitleList = subtitles;
            List<Subtitle> foundSubs = subtitles;
            var availableLang = subtitles.Select(item => item.Lang).Distinct().ToList();
            availableLang.Sort();
            subLangBindingSource.DataSource = availableLang;
            if (prefLang != null)
            {
                foundSubs = FilterSubs(subtitles, prefLang);
                if (foundSubs.Find(sub => sub.Lang.Contains(prefLang)) != null)
                {
                    comboBoxChangeLang.SelectedItem = prefLang;
                }
            }
            subtitleBindingSource.DataSource = foundSubs;
        }

        private List<Subtitle> FilterSubs(List<Subtitle> subtitles, string prefLang)
        {
            var filteredSubList = subtitles.FindAll(sub => sub.Lang.Contains(prefLang));
            if (filteredSubList.Count > 0)
            {
                return filteredSubList;
            } 
            return subtitles;
        }

        private void OpenSubtitles_Load(object sender, EventArgs e)
        {
        }

        private void DownloadButton_Click(object sender, EventArgs e)
        {
            try
            {
                using (new HourGlass())
                {
                    m_SelectedSub.Save();
                }
                Close();
            }
            catch (InternetConnectivityException)
            {
                MessageBox.Show(this, "MPDN was unable to access OpenSubtitles.org");
            }
            catch (Exception)
            {
                MessageBox.Show(this, "Can't download the selected subtitle.");
            }
        }

        private void dataGridView1_SelectionChanged(object sender, EventArgs e)
        {
            if (dataGridView1.SelectedRows.Count == 0)
                return;
            m_SelectedSub = (Subtitle) dataGridView1.SelectedRows[0].DataBoundItem;
        }

        private void dataGridView1_CellDoubleClick(object sender, DataGridViewCellEventArgs e)
        {
            DownloadButton_Click(sender, e);
        }

        private void comboBoxChangeLang_DropDownClosed(object sender, EventArgs e)
        {
            OpenSubtitlesLanguageHandler.ComboBoxPrefLanguageDropDownClosed(sender, e);
            var selectedLang = (string)comboBoxChangeLang.SelectedItem;
            var subs = FilterSubs(m_SubtitleList, selectedLang);
            subtitleBindingSource.DataSource = subs;
        }

    }
}
