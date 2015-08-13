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

using System;
using System.Linq;
using DirectShowLib;
using Mpdn.AudioScript;
using Mpdn.Extensions.Framework.Config;

namespace Mpdn.Extensions.Framework
{
    public abstract class AudioScript : AudioScript<NoSettings> { }

    public abstract class AudioScript<TSettings> : AudioScript<TSettings, ScriptConfigDialog<TSettings>>
        where TSettings : class, new()
    { }

    public abstract class AudioScript<TSettings, TDialog> : ExtensionUi<Config.Internal.AudioScripts, TSettings, TDialog>, IAudioScript
        where TSettings : class, new()
        where TDialog : ScriptConfigDialog<TSettings>, new()
    {
        protected abstract bool SupportBitStreaming { get; }
        protected abstract AudioSampleFormat[] SupportedSampleFormats { get; }
        protected abstract void Process(IntPtr samples, int length);

        #region Implementation

        public bool Process()
        {
            if (!SupportBitStreaming && Audio.InputFormat.IsBitStreaming())
                return false;

            if (!SupportedSampleFormats.Contains(Audio.InputFormat.SampleFormat()))
                return false;

            var input = Audio.Input;
            var output = Audio.Output;

            AudioHelpers.CopySample(input, output); // passthrough from input to output

            IntPtr samples;
            output.GetPointer(out samples);
            Process(samples, output.GetActualDataLength());

            return true;
        }

        public virtual void OnGetMediaType(WaveFormatExtensible format)
        {
            // Provides script a chance to change the output format
        }

        public virtual void OnNewSegment(long startTime, long endTime, double rate)
        {
        }

        #endregion
    }
}
