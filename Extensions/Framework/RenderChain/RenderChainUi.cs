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
using Mpdn.Extensions.Framework.Chain;
using Mpdn.Extensions.Framework.Config;
using Mpdn.RenderScript;

namespace Mpdn.Extensions.Framework.RenderChain
{
    public interface IRenderChainUi : IRenderScriptUi, IChainUi<ITextureFilter, IRenderScript> { }

    public static class RenderChainUi
    {
        public static readonly IRenderChainUi Identity = new IdentityRenderChainUi();

        public static bool IsIdentity(this IRenderChainUi chainUi)
        {
            return chainUi is IdentityRenderChainUi;
        }

        private class IdentityRenderChain : StaticChain<ITextureFilter>
        {
            public IdentityRenderChain() : base(x => x) { }
        }

        private class IdentityRenderChainUi : RenderChainUi<IdentityRenderChain>
        {
            public override string Category
            {
                get { return "Meta"; }
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

    public abstract class RenderChainUi<TChain> : RenderChainUi<TChain, ScriptConfigDialog<TChain>>
        where TChain : Chain<ITextureFilter>, new()
    { }

    public abstract class RenderChainUi<TChain, TDialog> : 
            ChainUi<ITextureFilter, IRenderScript, TChain, TDialog>,
            IRenderChainUi
        where TChain : Chain<ITextureFilter>, new()
        where TDialog : IScriptConfigDialog<TChain>, new()
    {
        public override IRenderScript CreateScript()
        {
            return new RenderChainScript(Settings);
        }
    }
}
