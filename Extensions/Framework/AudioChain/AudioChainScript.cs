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
using System.Diagnostics;
using DirectShowLib;
using Mpdn.AudioScript;
using Mpdn.Extensions.Framework.Chain;

namespace Mpdn.Extensions.Framework.AudioChain
{
    public class AudioChainScript : IAudioScript, IDisposable
    {
        protected readonly Chain<Audio> Chain;
        private IAudio m_Audio;

        public AudioChainScript(Chain<Audio> chain)
        {
            Chain = chain;
            AudioProc.Initialize();
        }

        ~AudioChainScript()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            Chain.Reset();
            m_Audio = null;

            if (disposing)
            {
                AudioProc.Destroy();
            }
        }

        public bool Execute()
        {
            try
            {
                var output = Chain.Process(new Audio(m_Audio.InputFormat, m_Audio.Input));
                AudioHelpers.CopySample(output.Sample, m_Audio.Output, true);
                return true;
            }
            catch (Exception ex)
            {
                Trace.WriteLine(ex);
                return false;
            }
        }

        public void OnGetMediaType(WaveFormatExtensible format)
        {
            // Provides script a chance to change the output format
        }

        public void Update(IAudio audio)
        {
            if (m_Audio != null)
            {
                throw new InvalidOperationException();
            }

            m_Audio = audio;
            Chain.Initialize();
        }
    }

    public struct Audio
    {
        public WaveFormatExtensible Format;
        public IMediaSample Sample;

        public Audio(WaveFormatExtensible format, IMediaSample sample)
        {
            Format = format;
            Sample = sample;
        }
    }
}
