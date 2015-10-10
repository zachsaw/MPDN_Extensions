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
using System.Collections.Generic;
using System.Linq;
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.Framework.Chain;
using Mpdn.Extensions.Framework.RenderChain;
using Mpdn.Extensions.PlayerExtensions;
using Mpdn.RenderScript;
using YAXLib;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Mpdn.ScriptGroup
    {
        public class ScriptGroup : PresetCollection<IFilter, IRenderScript>
        {
            #region Settings

            public int SelectedIndex { get; set; }

            public string Hotkey
            {
                get { return m_Hotkey; }
                set
                {
                    m_Hotkey = value ?? "";
                    UpdateHotkey();
                }
            }

            [YAXDontSerialize]
            public Preset<IFilter, IRenderScript> SelectedOption
            {
                get { return Options != null ? Options.ElementAtOrDefault(SelectedIndex) : null; }
            }

            [YAXDontSerialize]
            public string ScriptName
            {
                get
                {
                    var preset = SelectedOption;
                    return preset == null ? string.Empty : preset.Name; 
                }
                set
                {
                    var index = Options.FindIndex(p => p.Name == value);
                    if (index < 0)
                    {
                        throw new KeyNotFoundException(string.Format("ScriptName '{0}' is not found", value));
                    }
                    SelectedIndex = index;
                }
            }

            #endregion

            public ScriptGroup()
            {
                SelectedIndex = 0;
                m_HotkeyGuid = Guid.NewGuid();
            }

            public int GetPresetIndex(Guid guid)
            {
                return Options.FindIndex(o => o.Guid == guid);
            }

            public override IFilter Process(IFilter input)
            {
                return SelectedOption != null ? input + SelectedOption : input;
            }

            public override void Initialize()
            {
                RegisterHotkey();
                base.Initialize();
            }

            public override void Reset()
            {
                DeregisterHotkey();
                base.Reset();
            }

            #region Hotkey Handling

            private readonly Guid m_HotkeyGuid;
            private string m_Hotkey;
            private bool m_Registered;

            private void RegisterHotkey()
            {
                DynamicHotkeys.RegisterHotkey(m_HotkeyGuid, Hotkey, IncrementSelection);
                m_Registered = true;
            }

            private void DeregisterHotkey()
            {
                DynamicHotkeys.RemoveHotkey(m_HotkeyGuid);
                m_Registered = false;
            }

            private void UpdateHotkey()
            {
                if (m_Registered)
                {
                    DeregisterHotkey();
                    RegisterHotkey();
                }
            }

            private void IncrementSelection()
            {
                if (Options.Count > 0)
                {
                    SelectedIndex = (SelectedIndex + 1)%Options.Count;
                }

                if (SelectedOption != null)
                {
                    Player.OsdText.Show(Name + ": " + SelectedOption.Name);
                }

                Extension.SetRenderScript(Extension.RenderScriptGuid);
            }

            #endregion
        }

        public class ScriptGroupScript : RenderChainUi<ScriptGroup, ScriptGroupDialog>
        {
            protected override string ConfigFileName
            {
                get { return "Mpdn.ScriptGroup"; }
            }

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
                        Guid = new Guid("57D79B2B-4303-4102-A797-17EC4D003130"),
                        Name = "Script Group",
                        Description = Description()
                    };
                }
            }

            public string Description()
            {
                return (Settings.Options.Count > 0)
                    ? string.Join(", ",
                        Settings.Options.Select(x =>
                            (x == Settings.SelectedOption)
                                ? "[" + x.Name + "]"
                                : x.Name))
                    : "Picks one out of several renderscripts";
            }
        }
    }
}