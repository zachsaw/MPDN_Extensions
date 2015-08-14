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
using DirectShowLib;
using Mpdn.Extensions.Framework;

namespace Mpdn.Extensions.AudioScripts
{
    public class Reclock : Framework.AudioScript
    {
        private const double MAX_PERCENT_ADJUST = 2; // automatically reclock if the difference is less than 2%

        public override ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("51DA709D-7DDC-448B-BF0B-8691C9F39FF1"),
                    Name = "Reclock",
                    Description = "Reclock for MPDN (compatible with certain audio renderers only)"
                };
            }
        }

        public override bool Process()
        {
            if (Player.Filters.Video.Count == 0)
                return false;

            var input = Audio.Input;
            var output = Audio.Output;

            // passthrough from input to output
            AudioHelpers.CopySample(input, output, true);

            var stats = Player.Stats.Details;
            const int oneSecond = 1000000;
            var videoHz = oneSecond / stats.ActualSourceVideoIntervalUsec;
            var displayHz = oneSecond/stats.DisplayRefreshIntervalUsec;
            var ratio = displayHz / videoHz;
            if (ratio > (100 + MAX_PERCENT_ADJUST)/100 || ratio < (100 - MAX_PERCENT_ADJUST)/100)
                return true;

            var refclk = stats.RefClockDeviation;
            if (refclk > 10 || refclk < -10) // no data
            {
                refclk = 0;
            }

            // Use of 0.999999 is to allow a tiny amount of measurement error in displayHz
            // This allows us to adjust refclk to just a fraction under the displayHz
            var adjust = ratio*(0.999999 - refclk);
            long start, end;
            output.GetTime(out start, out end);
            long endDelta = end - start;
            start = (long)(start * adjust);
            output.SetTime(start, endDelta);

            return true;
        }

        public override void OnMediaClosed()
        {
        }

        public override void OnGetMediaType(WaveFormatExtensible format)
        {
        }

        public override void OnNewSegment(long startTime, long endTime, double rate)
        {
        }

        protected override void Process(float[,] samples, short channels, int sampleCount)
        {
            throw new InvalidOperationException();
        }
    }
}
