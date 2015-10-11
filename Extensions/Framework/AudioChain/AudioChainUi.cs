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
using Mpdn.Extensions.Framework.Config;
using YAXLib;

namespace Mpdn.Extensions.Framework.AudioChain
{
    public interface IAudioChainUi : IAudioScriptUi, IDisposable
    {
        AudioChain Chain { get; }
    }

    public abstract class AudioChainUi<TChain> : AudioChainUi<TChain, ScriptConfigDialog<TChain>>
        where TChain : AudioChain, new()
    { }

    public abstract class AudioChainUi<TChain, TDialog> : ExtensionUi<IAudioScript, TChain, TDialog>, IAudioChainUi
        where TChain : AudioChain, new()
        where TDialog : ScriptConfigDialog<TChain>, new()
    {
        protected AudioChainUi()
        {
            Settings = new TChain();
            CudafyInitializer.Init();
        }

        #region Implementation

        [YAXDontSerialize]
        public AudioChain Chain
        {
            get { return Settings; }
        }

        public IAudioScript CreateScript()
        {
            return new AudioChainScript(Settings);
        }

        #endregion Implementation

        #region GarbageCollecting

        ~AudioChainUi()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        public virtual void Dispose(bool disposing)
        {
            DisposeHelper.Dispose(Settings);
        }

        #endregion

        #region Implementation


        #endregion
    }
}
