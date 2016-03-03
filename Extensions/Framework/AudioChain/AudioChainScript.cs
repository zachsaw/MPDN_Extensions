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
using Mpdn.AudioScript;
using Mpdn.Extensions.Framework.Chain;

namespace Mpdn.Extensions.Framework.AudioChain
{
    public class AudioChainScript : FilterChainScript<IAudioFilter, IAudioOutput>, IAudioScript
    {
        private IAudio m_Audio;

        public AudioChainScript(Chain<IAudioFilter> chain) : base(chain)
        { }

        public void Update(IAudio audio)
        {
            if (!AudioProc.Initialize())
            {
                // Note: Using GuiThread.DoAsync ensures the warning gets displayed after the current media filename
                GuiThread.DoAsync(() =>
                    Player.OsdText.Show(
                        "Warning: Audio Script failed to initialize (Intel/AMD OpenCL Drivers unavailable)"));
                return;
            }

            m_Audio = audio;
            Update();
        }

        protected override void OutputResult(IAudioOutput result)
        {
            AudioHelpers.CopySample(result.Sample, m_Audio.Output, true);
        }

        protected override IAudioFilter MakeInitialFilter()
        {
            return new AudioSource(m_Audio);
        }

        protected override IAudioFilter HandleError(Exception e)
        {
            return new AudioSource(m_Audio);
        }
    }
}
