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
using System.Globalization;
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.Framework.Controls;
using Mpdn.Extensions.PlayerExtensions.Exceptions;
using MessageBox = System.Windows.Forms.MessageBox;

namespace Mpdn.Extensions.PlayerExtensions.Subtitles
{
    public class OpenSubtitlesExtension : PlayerExtension<OpenSubtitlesSettings, OpenSubtitlesConfigDialog>
    {
        private readonly OpenSubtitlesForm m_Form = new OpenSubtitlesForm();
        private readonly PlayerMenuItem m_MenuItem = new PlayerMenuItem(initiallyDisabled: true);

        private SubtitleDownloader m_Downloader;

        public override ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("ef5a12c8-246f-41d5-821e-fefdc442b0ea"),
                    Name = "OpenSubtitles",
                    Description = "Download automatically subtitles from OpenSubtitles"
                };
            }
        }

        public override IList<Verb> Verbs
        {
            get
            {
                return new[]
                {
                    new Verb(Category.View, string.Empty, "OpenSubtitles", "D", string.Empty, LaunchOpenSubtitleSearch, m_MenuItem)
                };
            }
        }

        private void LaunchOpenSubtitleSearch()
        {
            if (Player.State == PlayerState.Closed)
                return;

            Media.Pause();
            try
            {
                List<Subtitle> subList;
                using (new HourGlass())
                {
                    subList = m_Downloader.GetSubtitles(Media.FilePath);
                }
                if (subList == null || subList.Count == 0)
                {
                    MessageBox.Show(Gui.VideoBox, "No Subtitles found");
                    Media.Play();
                    return;
                }
                subList.Sort((a, b) => String.Compare(a.Lang, b.Lang, CultureInfo.CurrentUICulture, CompareOptions.StringSort));

                m_Form.SetSubtitles(subList, Settings.PreferedLanguage);
                m_Form.ShowDialog(Player.ActiveForm);
            }
            catch (InternetConnectivityException)
            {
                Trace.WriteLine("OpenSubtitles: Failed to access OpenSubtitles.org (InternetConnectivityException)");
            }
            catch (Exception)
            {
                Trace.WriteLine("OpenSubtitles: General exception occurred while trying to get subtitles");
            }
        }

        public override void Initialize()
        {
            base.Initialize();

            m_Downloader = new SubtitleDownloader("MPDN_Extensions");

            Player.StateChanged += PlayerStateChanged;
        }

        public override void Destroy()
        {
            Player.StateChanged -= PlayerStateChanged;

            base.Destroy();
        }

        private void PlayerStateChanged(object sender, PlayerStateEventArgs args)
        {
            m_MenuItem.Enabled = args.NewState != PlayerState.Closed;
        }
    }

    public class OpenSubtitlesSettings
    {
        public OpenSubtitlesSettings()
        {
            PreferedLanguage = CultureInfo.CurrentUICulture.Parent.EnglishName;
        }

        public string PreferedLanguage { get; set; }
    }
}
