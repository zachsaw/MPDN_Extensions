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
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.Framework.Controls;
using Mpdn.Extensions.PlayerExtensions.Exceptions;

namespace Mpdn.Extensions.PlayerExtensions
{
    public partial class OpenSubtitlesForm : Form
    {
        private Subtitle m_SelectedSub;
        private List<Subtitle> m_SubtitleList; 

        public OpenSubtitlesForm()
        {
            InitializeComponent();
            gridView.AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.DisplayedCells;
            CancelButton = btnCancel;
            Icon = Gui.Icon;
            FormClosed += OpenSubtitlesFormFormClosed;
            comboBoxChangeLang.KeyDown += OpenSubtitlesLanguageHandler.ComboBoxPrefLanguageKeyDown;
            comboBoxChangeLang.KeyPress += OpenSubtitlesLanguageHandler.ComboBoxPrefLanguageKeyPress;
            comboBoxChangeLang.DropDownClosed += OpenSubtitlesLanguageHandler.ComboBoxPrefLanguageDropDownClosed;
        }

        private void OpenSubtitlesFormFormClosed(object sender, FormClosedEventArgs e)
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

        private void DownloadButtonClick(object sender, EventArgs e)
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

        private void GridViewSelectionChanged(object sender, EventArgs e)
        {
            if (gridView.SelectedRows.Count == 0)
                return;
            m_SelectedSub = (Subtitle) gridView.SelectedRows[0].DataBoundItem;
        }

        private void GridViewCellDoubleClick(object sender, DataGridViewCellEventArgs e)
        {
            DownloadButtonClick(sender, e);
        }

        private void LanguageSelectedValueChanged(object sender, EventArgs e)
        {
            var selectedLang = (string)comboBoxChangeLang.SelectedItem;
            if (selectedLang != null) {
                var subs = FilterSubs(m_SubtitleList, selectedLang);
                subtitleBindingSource.DataSource = subs;
            }
        }

    }
}
