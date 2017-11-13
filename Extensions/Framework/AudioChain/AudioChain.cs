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
using Mpdn.Extensions.Framework.Chain.Dialogs;

namespace Mpdn.Extensions.Framework.AudioChain
{
    public class AudioChain<TProcess> : Chain<IAudioFilter> //PinFilterChain<TFilter, IAudioFilter>
        where TProcess : class, IAudioProcess, new()
    {
        public TProcess FilterProcess { get; set; }

        public AudioChain()
        {
            FilterProcess = new TProcess();
        }

        public override IAudioFilter Process(IAudioFilter input)
        {
            return ConfigHelper.MakeXMLDuplicate(FilterProcess).ApplyTo(input);
        }
    }

    public class AudioScriptChain : ScriptChain<IAudioFilter, IAudioScript> { }
    public class AudioScriptChainDialog : ScriptChainDialog<IAudioFilter, IAudioScript> { }

    public class AudioScriptGroup : ScriptGroup<IAudioFilter, IAudioScript> { }
    public class AudioScriptGroupDialog : ScriptGroupDialog<IAudioFilter, IAudioScript> { }
}
