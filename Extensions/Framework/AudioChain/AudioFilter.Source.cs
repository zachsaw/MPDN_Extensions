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
using Mpdn.Extensions.Framework.Filter;

namespace Mpdn.Extensions.Framework.AudioChain
{
    using static FilterBaseHelper;
    using static FilterOutputHelper;

    public class SourceAudioOutput : FilterOutput<IAudioOutput>, IAudioOutput
    {
        private readonly IAudio m_Audio;

        public SourceAudioOutput(IAudio audio)
        {
            m_Audio = audio;
        }

        public WaveFormatExtensible Format
        {
            get { return m_Audio.InputFormat; }
        }

        public IMediaSample MediaSample
        {
            get { return m_Audio.Input; }
        }

        public IMediaSample Sample
        {
            get { return m_Audio.Input; }
        }

        public IAudioDescription Output { get { return this; } }

        protected override IAudioOutput Value { get { return this; } }

        protected override void Allocate() { }
        protected override void Deallocate() { }
    }

    public class AudioSource : AudioFilter
    {
        public AudioSource(IAudio audio) : this(new SourceAudioOutput(audio)) { }

        private AudioSource(IAudioOutput audioOutput)
            : base(Return(Return(audioOutput, audioOutput)))
        { }
    }
}
