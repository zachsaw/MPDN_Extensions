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
using System.IO;
using Mpdn.Extensions.PlayerExtensions.Exceptions;

namespace Mpdn.Extensions.PlayerExtensions
{
    public class OpenSubtitlesExtension : PlayerExtension<OpenSubtitlesSettings, OpenSubtitlesConfigDialog>
    {
        private SubtitleDownloader m_Downloader;
        private readonly OpenSubtitlesForm m_Form = new OpenSubtitlesForm();

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

        public override void Initialize()
        {
            base.Initialize();
            m_Downloader = new SubtitleDownloader("MPDN_Extensions");
            PlayerControl.MediaLoading += MediaLoading;
        }

        public override void Destroy()
        {
            base.Destroy();
            PlayerControl.MediaLoading -= MediaLoading;
        }


        private void MediaLoading(object sender, MediaLoadingEventArgs e)
        {
            if (!Settings.EnableAutoDownloader)
                return;
            if (HasExistingSubtitle(e.Filename))
                return;
            try
            {
                List<Subtitle> subList;
                using (new HourGlass())
                {
                    subList = m_Downloader.GetSubtitles(e.Filename);
                }
                if (subList == null || subList.Count == 0)
                    return; // Opensubtitles messagebox is annoying #44 https://github.com/zachsaw/MPDN_Extensions/issues/44
                subList.Sort((a, b) => String.Compare(a.Lang, b.Lang, CultureInfo.CurrentUICulture, CompareOptions.StringSort));

                m_Form.SetSubtitles(subList, Settings.PreferedLanguage);
                m_Form.ShowDialog(PlayerControl.Form);
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

        private bool HasExistingSubtitle(string mediaFilename)
        {
            var subFile = string.Format(Subtitle.FILE_NAME_FORMAT, Path.GetFileNameWithoutExtension(mediaFilename),
                Settings.PreferedLanguage);
            var fullPath = Path.Combine(PathHelper.GetDirectoryName(mediaFilename), subFile);
            return File.Exists(fullPath);
        }

        public override IList<Verb> Verbs
        {
            get { return new Verb[0]; }
        }
    }

    public class OpenSubtitlesSettings
    {
        public OpenSubtitlesSettings()
        {
            EnableAutoDownloader = false;
            PreferedLanguage = CultureInfo.CurrentUICulture.Parent.EnglishName;
        }

        public bool EnableAutoDownloader { get; set; }
        public string PreferedLanguage { get; set; }
    }
}
