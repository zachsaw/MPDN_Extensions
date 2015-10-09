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
using DirectShowLib;
using Mpdn.AudioScript;

namespace Mpdn.Extensions.Framework.AudioChain
{
    public class AudioChainScript : IAudioScript, IDisposable
    {
        protected readonly AudioChain Chain;

        public AudioChainScript(AudioChain chain)
        {
            Chain = chain;
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
        }

        public bool Execute()
        {
            return Chain.Process();
        }

        public void OnGetMediaType(WaveFormatExtensible format)
        {
            // Provides script a chance to change the output format
        }

        public void Update(IAudio audio)
        {
            Chain.Initialize(audio);
        }
    }

    public struct AudioParam
    {
        public WaveFormatExtensible Format;
        public IMediaSample Sample;

        public AudioParam(WaveFormatExtensible format, IMediaSample sample)
        {
            Format = format;
            Sample = sample;
        }
    }
}
