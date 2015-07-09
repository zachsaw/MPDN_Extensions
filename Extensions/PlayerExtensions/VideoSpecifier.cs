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
using System.Text.RegularExpressions;
using DirectShowLib;
using Mpdn.Extensions.Framework;

namespace Mpdn.Extensions.PlayerExtensions
{
    public class VideoSpecifier
    {
        public static bool Match(string specifier)
        {
            if (!IsValid(specifier))
                return false;

            var vid = Media.VideoInfo.BmiHeader;

            var regexWidth = new Regex(@"w(\d+)");
            var widthMatch = regexWidth.Match(specifier).Groups[1];
            var width = widthMatch.Success ? int.Parse(widthMatch.Value) : vid.Width;

            var regexHeight = new Regex(@"h(\d+)");
            var heightMatch = regexHeight.Match(specifier).Groups[1];
            var height = heightMatch.Success ? int.Parse(heightMatch.Value) : vid.Height;

            var vidIsInterlaced = Media.VideoInfo.InterlaceFlags.HasFlag(AMInterlace.IsInterlaced);
            var vidFps = 1000000/(int) Media.VideoInfo.AvgTimePerFrame;

            bool interlaced = vidIsInterlaced;
            int frameRate = vidFps;

            var regexFrameRate = new Regex(@"(i|p)(\d+)");
            var frameRateMatch = regexFrameRate.Match(specifier);
            if (frameRateMatch.Success)
            {
                interlaced = frameRateMatch.Groups[1].Value == "i";
                frameRate = int.Parse(frameRateMatch.Groups[2].Value);
            }

            return width == vid.Width && height == vid.Height && interlaced == vidIsInterlaced && frameRate == vidFps;
        }

        public static bool IsValid(string vt)
        {
            var regexValidate = new Regex(@"^(w{1}\d+)?(h{1}\d+)?((i|p){1}\d+)?$");
            return regexValidate.Matches(vt).Count != 0;
        }

        public static string FormatHelp
        {
            get { return "[w (0..9)] [h (0..9)] [p | i (0..9)]"; }
        }

        public static string ExampleHelp
        {
            get { return "w1920p24 p30 w320h180"; }
        }
    }
}