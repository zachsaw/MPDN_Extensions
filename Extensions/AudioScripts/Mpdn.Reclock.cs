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
using Mpdn.Extensions.Framework;

namespace Mpdn.Extensions.AudioScripts
{
    namespace Mpdn
    {
        public class Reclock : Framework.AudioScript
        {
            private const double MAX_PERCENT_ADJUST = 3; // automatically reclock if the difference is less than 3%
            private const double SANEAR_OVERSHOOT = 15;
            private const double DIRECTSOUND_OVERSHOOT = 10;
            private const double MAX_SWING = 0.0005;

            private static readonly Guid s_SanearSoundClsId = new Guid("DF557071-C9FD-433A-9627-81E0D3640ED9");
            private static readonly Guid s_DirectSoundClsId = new Guid("79376820-07D0-11CF-A24D-0020AFD79767");
            private static readonly Guid s_WaveOutClsId = new Guid("E30629D1-27E5-11CE-875D-00608CB78066");

            private bool m_Sanear;
            private bool m_DirectSoundWaveOut;

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

            public override bool Process(AudioParam input, AudioParam output)
            {
                if (!CompatibleAudioRenderer)
                    return false;

                var stats = Player.Stats.Details;
                if (stats == null)
                    return false;

                if (stats.ActualSourceVideoIntervalUsec < 1e-8)
                    return false; // audio only - no need to reclock

                const int oneSecond = 1000000;
                var videoHz = oneSecond/stats.ActualSourceVideoIntervalUsec;
                var displayHz = oneSecond/stats.DisplayRefreshIntervalUsec;
                var ratio = displayHz/videoHz;
                if (ratio > (100 + MAX_PERCENT_ADJUST)/100 || ratio < (100 - MAX_PERCENT_ADJUST)/100)
                    return false;

                // passthrough from input to output
                AudioHelpers.CopySample(input.Sample, output.Sample, true);

                var refclk = stats.RefClockDeviation;
                if (refclk > -10 && refclk < 10)
                {
                    // Fine tune further when we have refclk stats
                    var actualVideoHz = videoHz*(1 + refclk);
                    var finalDifference = displayHz/actualVideoHz;
                    // finalDifference will get smaller over time as refclk inches closer and closer to the target value
                    var overshoot = m_Sanear ? SANEAR_OVERSHOOT : DIRECTSOUND_OVERSHOOT;
                    var swing = Math.Pow(finalDifference, overshoot);
                    ratio *= Math.Max(1-MAX_SWING, Math.Min(1+MAX_SWING, swing));
                    // Let it overshoot the final difference so we can converge faster
                    // Sanear has a built-in low-pass filter that allows it to creep its sample rate towards the target
                    // so we need a much higher overshoot to make it converge faster
                    // DSound on the other hand doesn't
                }

                long start, end;
                output.Sample.GetTime(out start, out end);
                long endDelta = end - start;
                start = (long) (start*ratio);
                output.Sample.SetTime(start, endDelta);

                return true;
            }

            public bool CompatibleAudioRenderer
            {
                get { return m_Sanear || m_DirectSoundWaveOut; }
            }

            public override void OnMediaClosed()
            {
                m_Sanear = false;
                m_DirectSoundWaveOut = false;
            }

            public override void OnGetMediaType(WaveFormatExtensible format)
            {
                GuiThread.DoAsync(delegate
                {
                    // This has to be done via GuiThread.DoAsync because when this method is called
                    // Player.Filters has not been populated
                    // Using GuiThread.DoAsync essentially queues this delegate until the media file
                    // is actually opened and all Player fields have been populated
                    if (Player.Filters.FirstOrDefault(f => f.ClsId == s_SanearSoundClsId) != null)
                    {
                        m_Sanear = true;
                    }
                    else if (
                        Player.Filters.FirstOrDefault(
                            f => f.ClsId == s_DirectSoundClsId || f.ClsId == s_WaveOutClsId) != null)
                    {
                        m_DirectSoundWaveOut = true;
                    }
                    else
                    {
                        Player.OsdText.Show("Warning: Audio renderer is incompatible with Reclock. Reclock disabled!");
                    }
                });
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
}