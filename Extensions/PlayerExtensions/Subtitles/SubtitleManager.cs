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
using System.Diagnostics;
using System.IO;
using System.Linq;
using DirectShowLib;
using Mpdn.DirectShow;
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.PlayerExtensions.Interfaces;

namespace Mpdn.Extensions.PlayerExtensions.Subtitles
{
    public static class SubtitleManager
    {
        [Flags]
        private enum SubType
        {
            SRT = 0,
            SUB,
            SMI,
            PSB,
            SSA,
            ASS,
            IDX,
            USF,
            XSS,
            TXT,
            RT,
            SUP
        };
        public class SubtitleTiming
        {
            public int Delay { get; set; }
            public int SpeedMultiplier { get; set; }
            public int SpeedDivisor { get; set; }

            public SubtitleTiming()
            {
            }

            public SubtitleTiming(int delay, int speedMultiplier, int speedDivisor)
            {
                Delay = delay;
                SpeedMultiplier = speedMultiplier;
                SpeedDivisor = speedDivisor;
            }
        }
        private static readonly Guid s_XySubFilterGuid = new Guid("2dfcb782-ec20-4a7c-b530-4577adb33f21");
        private static readonly Guid s_VSFilterGuid = new Guid("9852a670-f845-491b-9be6-ebd841b8a613");

        public static Filter SubtitleFilter
        {
            get
            {
                return Player.Filters.FirstOrDefault(f => f.ClsId == s_XySubFilterGuid || f.ClsId == s_VSFilterGuid);
            }
        }

        public static bool IsSubtitleFilterLoaded()
        {
            return SubtitleFilter != null;
        }

        public static SubtitleTiming GetTiming()
        {
            if (!IsSubtitleFilterLoaded())
                return null;

            SubtitleTiming timing = null;
            ComThread.Do(() =>
            {
                var extSubSource = SubtitleFilter.Base as IDirectVobSub;
                int delay, mul, div;
                var hr = extSubSource.get_SubtitleTiming(out delay, out mul, out div);
                DsError.ThrowExceptionForHR(hr);
                timing = new SubtitleTiming(delay, mul, div);
            });
            return timing;
        } 

        public static bool SetTiming(SubtitleTiming timing)
        {
            if (!IsSubtitleFilterLoaded())
                return false;

            ComThread.Do(() =>
            {
                var extSubSource = SubtitleFilter.Base as IDirectVobSub;
                var hr = extSubSource.put_SubtitleTiming(timing.Delay, timing.SpeedMultiplier, timing.SpeedDivisor);
                DsError.ThrowExceptionForHR(hr);
            });
            return true;
        }

        public static bool LoadFile(string subtitleFile)
        {
            if (!IsSubtitleFilterLoaded())
                return false;

            ComThread.Do(() =>
            {
                var extSubSource = SubtitleFilter.Base as IDirectVobSub;
                var hr = extSubSource.put_FileName(subtitleFile);
                DsError.ThrowExceptionForHR(hr);
            });
            return true;
        }

        public static bool SelectLanguage(string lang)
        {
            ComThread.Do(() =>
            {
                var extSubSource = SubtitleFilter.Base as IDirectVobSub;
                if (extSubSource == null || string.IsNullOrWhiteSpace(lang))
                    return;

                int iCount;

                var hr = extSubSource.get_LanguageCount(out iCount);
                DsError.ThrowExceptionForHR(hr);
                Trace.WriteLine("LoadExternalSubtitle Count: " + iCount);

                for (var i = 0; i < iCount; i++)
                {
                    string langName;

                    hr = extSubSource.get_LanguageName(i, out langName);
                    DsError.ThrowExceptionForHR(hr);

                    Trace.WriteLine("LoadExternalSubtitle SubName " + langName);

                    if (lang != langName)
                        continue;

                    Trace.WriteLine("LoadExternalSubtitle Select Stream " + i);

                    hr = extSubSource.put_SelectedLanguage(i);
                    DsError.ThrowExceptionForHR(hr);

                    var iSelected = 0;
                    hr = extSubSource.get_SelectedLanguage(out iSelected);
                    DsError.ThrowExceptionForHR(hr);

                    Trace.WriteLine("LoadExternalSubtitle Select Result: " + iSelected);

                    break;
                }
            });
            return true;
        }
        /// <summary>
        /// Check if the extension of the file is linked to a Subtitle format.
        /// </summary>
        /// <param name="file"></param>
        /// <returns></returns>
        public static bool IsSubtitleFile(string file)
        {
            var extension = Path.GetExtension(file);

            if (extension == null)
                return false;

            var ext = extension.Remove(0,1);
            return Enum.IsDefined(typeof (SubType), ext.ToUpper());
        }
    }
}