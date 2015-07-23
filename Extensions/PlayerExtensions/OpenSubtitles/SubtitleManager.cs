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
using System.Linq;
using DirectShowLib;
using Mpdn.DirectShow;
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.PlayerExtensions.Interfaces;

namespace Mpdn.Extensions.PlayerExtensions.OpenSubtitles
{
    public static class SubtitleManager
    {
        private static readonly Guid s_XyDirectVobSub = new Guid("2dfcb782-ec20-4a7c-b530-4577adb33f21");
        private static Filter s_DirectVobFilter;

        public static Filter DirectVobSubFilter
        {
            get
            {
                if (s_DirectVobFilter != null)
                {
                    return s_DirectVobFilter;
                }
                s_DirectVobFilter = Player.Filters.Video.FirstOrDefault(f => f.ClsId == s_XyDirectVobSub);

                return s_DirectVobFilter;
            }
        }

        public static bool IsSubtitleFilterLoaded()
        {
            return DirectVobSubFilter != null;
        }

        public static bool LoadSubtitleFile(string filepath)
        {
            if (!IsSubtitleFilterLoaded())
                return false;

            var result = -1;
            ComThread.Do(() =>
            {
                var vobFilter = (IDirectVobSub) DirectVobSubFilter.Base;
                result = vobFilter.put_FileName(filepath);
            });
            return result == 0;
        }

    }
}