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
using Mpdn.Extensions.Framework.Filter;

namespace Mpdn.Extensions.Framework.AudioChain
{
    public abstract class AudioConfigDialog<TProcess> : ScriptConfigDialog<AudioChain<TProcess>> 
        where TProcess : class, IAudioProcess, new()
    {
        new protected TProcess Settings { get { return base.Settings.FilterProcess; } }
    }

    public interface IAudioChainUi : IAudioScriptUi, IChainUi<IAudioFilter, IAudioScript> { }

    public abstract class AudioChainUi<TProcess> : AudioChainUi<TProcess, ScriptConfigDialog<AudioChain<TProcess>>>
        where TProcess : class, IAudioProcess, new()
    { }

    public abstract class AudioChainUi<TProcess, TDialog> :
            GeneralAudioChainUI<AudioChain<TProcess>, TDialog>, 
            IAudioChainUi
        where TProcess : class, IAudioProcess, new()
        where TDialog : IScriptConfigDialog<AudioChain<TProcess>>, new()
    {
        public override void Initialize()
        {
            base.Initialize();

            //CudafyInitializer.Init();
            AudioProc.AsyncInitialize();
        }
    }

    public abstract class GeneralAudioChainUI<TChain, TDialog> :
            ChainUi<IAudioFilter, IAudioScript, TChain, TDialog>, 
            IAudioChainUi
        where TChain : Chain<IAudioFilter>, new()
        where TDialog : IScriptConfigDialog<TChain>, new()
    {
        public override IAudioScript CreateScript()
        {
            return new AudioChainScript(Settings);
        }
    }
}
