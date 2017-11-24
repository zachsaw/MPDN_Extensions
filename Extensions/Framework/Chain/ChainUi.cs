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
using Mpdn.Extensions.Framework.Config;
using YAXLib;

namespace Mpdn.Extensions.Framework.Chain
{
    public interface IChainUi<T, out TScript> : IScriptUi<TScript>
        where TScript : class, IScript
    {
        Chain<T> Chain { get; }
        string Category { get; }
    }

    public static class ChainUiHelper
    {
        public static bool IsIdentity<T, TScript>(this IChainUi<T, TScript> chainUi)
            where TScript : class, IScript
        {
            return chainUi == ChainUi<T, TScript>.IDENTITY;
        }
    }

    public static class ChainUi<T, TScript>
        where TScript : class, IScript
    {
        public static readonly IChainUi<T, TScript> IDENTITY = new IdentityChainUi();

        private class IdentityChain : StaticChain<T>
        {
            public IdentityChain() : base(x => x) { }
        }

        private class IdentityChainUi : ChainUi<T, TScript, IdentityChain>
        {
            public override string Category
            {
                get { return "Meta"; }
            }

            public override TScript CreateScript()
            {
                return null;
            }

            public override ExtensionUiDescriptor Descriptor
            {
                get
                {
                    return new ExtensionUiDescriptor
                    {
                        Guid = Guid.Empty,
                        Name = "None",
                        Description = "Do nothing"
                    };
                }
            }
        }
    }

    public abstract class ChainUi<T, TScript, TChain> : ChainUi<T, TScript, TChain, ScriptConfigDialog<TChain>>
        where TScript : class, IScript
        where TChain : Chain<T>, new()
    { }

    public abstract class ChainUi<T, TScript, TChain, TDialog> : 
            ExtensionUi<TScript, TChain, TDialog>,
            IChainUi<T, TScript>
        where TScript : class, IScript
        where TChain : Chain<T>, new()
        where TDialog : IScriptConfigDialog<TChain>, new()
    {
        public abstract string Category { get; }

        public abstract TScript CreateScript();

        protected ChainUi()
        {
            Settings = new TChain();
        }

        [YAXDontSerialize]
        public Chain<T> Chain
        {
            get { return Settings; }
        }
    }
}
