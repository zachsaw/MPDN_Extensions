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

using Mpdn.AudioScript;
using Mpdn.Extensions.Framework.Chain;
using Mpdn.Extensions.Framework.Config;

namespace Mpdn.Extensions.Framework.AudioChain
{
    public interface IAudioChainUi : IAudioScriptUi, IChainUi<Audio, IAudioScript> { }

    public abstract class AudioChainUi<TChain> : AudioChainUi<TChain, ScriptConfigDialog<TChain>>
        where TChain : Chain<Audio>, new()
    { }

    public abstract class AudioChainUi<TChain, TDialog> : 
            ChainUi<Audio, IAudioScript, TChain, TDialog>, 
            IAudioChainUi
        where TChain : Chain<Audio>, new()
        where TDialog : IScriptConfigDialog<TChain>, new()
    {
        protected AudioChainUi()
        {
            CudafyInitializer.Init();
        }

        public override IAudioScript CreateScript()
        {
            return new AudioChainScript(Settings);
        }
    }
}
