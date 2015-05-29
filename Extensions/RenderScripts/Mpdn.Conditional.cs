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
using System.Linq;
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.RenderScripts.Mpdn.Presets;
using Mpdn.Extensions.RenderScripts.Mpdn.ScriptedRenderChain;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Mpdn.Conditional
    {
        public class Conditional : RenderChain
        {
            #region Settings

            public string Condition { get; set; }
            public Guid Preset { get; set; }

            #endregion

            private MpdnScriptEngine m_Engine;

            public override void Initialize()
            {
                m_Engine = new MpdnScriptEngine();
                base.Initialize();
            }

            public override void Reset()
            {
                DisposeHelper.Dispose(ref m_Engine);
                base.Reset();
            }

            public override IFilter CreateFilter(IFilter input)
            {
                if (string.IsNullOrWhiteSpace(Condition) || Preset == Guid.Empty)
                    return input;

                return m_Engine.Execute(this, input, GetScript(), "Conditional");
            }

            private string GetScript()
            {
                var scriptGroupScript = new ScriptGroupScript().CreateNew(true);
                var presetGroup = (PresetGroup) scriptGroupScript.Chain;

                var preset = presetGroup.GetPreset(Preset);
                if (preset == null)
                {
                    throw new Exception("Preset is not found!");
                }
                return string.Format(
                    "if ({0}) {{ var s = Script.LoadByClassName(\"PresetGroup\"); s.ScriptName = \"{1}\"; input.Add(s); }}",
                    Parser.BuildCondition(Condition), preset.Name);
            }
        }

        public class ConditionalUi : RenderChainUi<Conditional, ConditionalConfigDialog>
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
                        Name = "Conditional",
                        Description = GetDescription(),
                        Guid = new Guid("8ADCC25B-BAB0-4C88-AE37-695182048815"),
                        Copyright = "" // Optional field
                    };
                }
            }

            private string GetDescription()
            {
                if (!string.IsNullOrWhiteSpace(Settings.Condition) && Settings.Preset != Guid.Empty)
                {
                    var scriptGroupScript = new ScriptGroupScript().CreateNew(true);
                    var presetGroup = (PresetGroup) scriptGroupScript.Chain;
                    var preset = presetGroup.GetPreset(Settings.Preset);
                    if (preset != null)
                    {
                        return string.Format("If {0}, use preset '{1}'", Settings.Condition.Trim(), preset.Name);
                    }
                }
                return "Conditionally activates a preset";
            }
        }
    }
}
