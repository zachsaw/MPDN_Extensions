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

using System.IO;
using Mpdn.Extensions.Framework.Chain;
using Mpdn.Extensions.Framework.Chain.Dialogs;
using System;
using Mpdn.OpenCl;
using Mpdn.RenderScript;

namespace Mpdn.Extensions.Framework.RenderChain
{
    public abstract class RenderChain : FilterChain<ITextureFilter>
    {
        protected RenderChain()
        {
            ShaderCache.Load();
        }
    
        #region Shader Compilation

        protected virtual string ShaderPath
        {
            get { return GetType().Name; }
        }

        protected string ShaderDataFilePath
        {
            get { return Path.Combine(ShaderCache.ShaderPathRoot, ShaderPath); }
        }

        protected ShaderFromFile FromFile(string shaderFileName, string profile = "ps_3_0", string entryPoint = "main", string compilerOptions = null)
        {
            return DefinitionHelper.FromFile(Path.Combine(ShaderDataFilePath, shaderFileName), profile, entryPoint, compilerOptions);
        }

        protected ShaderFromString FromString(string shadercode, string profile = "ps_3_0", string entryPoint = "main", string compilerOptions = null)
        {
            return DefinitionHelper.FromString(shadercode, profile, entryPoint, compilerOptions);
        }

        protected ShaderFromByteCode FromByteCode(string bytecodeFileName)
        {
            return DefinitionHelper.FromByteCode(Path.Combine(ShaderDataFilePath, bytecodeFileName));
        }

        #endregion
    }

    public class RenderScriptChain : ScriptChain<ITextureFilter, IRenderScript> { }
    public class RenderScriptChainDialog : ScriptChainDialog<ITextureFilter, IRenderScript> { }

    public class RenderScriptGroup : ScriptGroup<ITextureFilter, IRenderScript> { }
    public class RenderScriptGroupDialog : ScriptGroupDialog<ITextureFilter, IRenderScript> { }
}