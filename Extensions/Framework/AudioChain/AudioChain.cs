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
using Mpdn.Extensions.Framework.Chain.Dialogs;
using Mpdn.Extensions.Framework.Filter;

namespace Mpdn.Extensions.Framework.AudioChain
{
    public class AudioChain<TFilter> : PinFilterChain<TFilter, IAudioFilter>
        where TFilter : PinFilter<IAudioOutput>, IAudioFilter, new()
    {
        public TFilter FilterSettings { get; set; }

        public AudioChain()
        {
            FilterSettings = new TFilter();
        }

        protected override TFilter MakeFilter()
        {
            return ConfigHelper.MakeXMLDuplicate<TFilter>(FilterSettings);
        }
    }

    public class AudioScriptChain : ScriptChain<IAudioFilter, IAudioScript> { }
    public class AudioScriptChainDialog : ScriptChainDialog<IAudioFilter, IAudioScript> { }

    public class AudioScriptGroup : ScriptGroup<IAudioFilter, IAudioScript> { }
    public class AudioScriptGroupDialog : ScriptGroupDialog<IAudioFilter, IAudioScript> { }
}
