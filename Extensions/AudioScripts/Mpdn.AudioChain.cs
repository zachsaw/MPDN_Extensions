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
using Mpdn.Extensions.Framework;
using DirectShowLib;

namespace Mpdn.Extensions.AudioScripts
{
    namespace Mpdn
    {
        public class AudioChainSettings
        {
            public List<IExtensionUi> AudioScripts { get; set; }

            public AudioChainSettings()
            {
                AudioScripts = new List<IExtensionUi>();
            }
        }

        public class AudioChain : AudioScript<AudioChainSettings, AudioChainConfigDialog>
        {
            public override ExtensionUiDescriptor Descriptor
            {
                get
                {
                    return new ExtensionUiDescriptor
                    {
                        Guid = new Guid("92E3FD70-C663-4C94-9B17-2EC10D9EA8CC"),
                        Name = "AudioChain",
                        Description = "Audio script chaining"
                    };
                }
            }

            public override bool Process()
            {
                var input = new AudioParam(Audio.InputFormat, Audio.Input);
                var output = new AudioParam(Audio.OutputFormat, Audio.Output);
                var first = AudioScripts.FirstOrDefault();
                if (first == null)
                    return false;

                if (!first.Process(input, output))
                    return false;

                var second = AudioScripts.Skip(1).FirstOrDefault();
                if (second == null)
                    return true;

                using (var tempSample = new MediaSample(output.Sample))
                {
                    var temp = new AudioParam(Audio.OutputFormat, tempSample);
                    if (!second.Process(output, temp))
                        return true;

                    foreach (var s in AudioScripts.Skip(2))
                    {
                        if (!s.Process(temp, input))
                            return false;

                        Swap(ref temp, ref input);
                    }

                    AudioHelpers.CopySample(temp.Sample, output.Sample, true);
                }

                return true;
            }

            private static void Swap(ref AudioParam p1, ref AudioParam p2)
            {
                var temp = p1;
                p1 = p2;
                p2 = temp;
            }

            public override void OnMediaClosed()
            {
                foreach (var audioScript in AudioScripts)
                {
                    audioScript.OnMediaClosed();
                }
            }

            public override void OnGetMediaType(WaveFormatExtensible format)
            {
                foreach (var audioScript in AudioScripts)
                {
                    audioScript.OnGetMediaType(format);
                }
            }

            public override void OnNewSegment(long startTime, long endTime, double rate)
            {
                foreach (var audioScript in AudioScripts)
                {
                    audioScript.OnNewSegment(startTime, endTime, rate);
                }
            }

            protected override void Process(float[,] samples, short channels, int sampleCount)
            {
                throw new InvalidOperationException();
            }

            private IEnumerable<IAudioChain> AudioScripts
            {
                get { return Settings.AudioScripts.OfType<IAudioChain>(); }
            }
        }
    }
}
