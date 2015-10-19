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
// 

using System;
using System.IO;
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.Framework.RenderChain;
using Mpdn.Extensions.Framework.Scripting;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Mpdn.ScriptedRenderChain
    {
        using Engine = ScriptEngine<RenderChain>;

        public class ScriptedRenderChain : RenderChain
        {
            #region Settings

            public string ScriptFileName { get; set; }

            #endregion

            private ScriptParser m_ScriptParser;
            private string m_RsFile;
            private string m_RsFileName;
            private DateTime m_LastModified = DateTime.MinValue;

            public ScriptedRenderChain()
            {
                if (!string.IsNullOrWhiteSpace(ScriptFileName)) return;
                ScriptFileName = Helpers.DefaultScriptFileName;
                if (!File.Exists(ScriptFileName))
                {
                    CreateDefaultScriptFile();
                }
            }

            public override void Initialize()
            {
                m_ScriptParser = new ScriptParser(Engine.FilterTypeNames);
                base.Initialize();
            }

            private void CreateDefaultScriptFile()
            {
                File.WriteAllText(ScriptFileName, Helpers.DefaultScript);
            }

            protected override IFilter CreateFilter(IFilter input)
            {
                var clip = Engine.Execute(input, BuildScript(ScriptFileName), ScriptFileName);
                if (clip == null)
                    return null;

                return clip.Filter;
            }

            private string BuildScript(string scriptRs)
            {
                scriptRs = Path.GetFullPath(scriptRs);

                var lastMod = File.GetLastWriteTimeUtc(scriptRs);
                if (m_RsFileName == scriptRs && lastMod == m_LastModified)
                    return m_RsFile;

                m_RsFile = m_ScriptParser.BuildScript(File.ReadAllText(scriptRs));
                m_RsFileName = scriptRs;
                m_LastModified = lastMod;

                return m_RsFile;
            }
        }

        public class ScriptedRenderChainUi : RenderChainUi<ScriptedRenderChain, ScriptedRenderChainConfigDialog>
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
                        Name = "Scripted Render Chain",
                        Description = "Write your own render chain using Avisynth-like scripting language",
                        Guid = new Guid("E38CC06E-F1EB-4D57-A01B-C7010D0D9C6A"),
                        Copyright = "" // Optional field
                    };
                }
            }
        }
    }
}
