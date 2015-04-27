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
using System.Collections.Generic;
using System.Linq;
using System.Windows.Forms;
using Mpdn.PlayerExtensions;
using YAXLib;

namespace Mpdn.RenderScript
{
    namespace Mpdn.Presets
    {
        public class Preset : RenderChain
        {
            [YAXAttributeForClass]
            public virtual string Name { get; set; }

            [YAXAttributeForClass]
            public virtual Guid Guid { get; set; }

            [YAXDontSerialize]
            public virtual string Description { get { return null; } }

            public virtual bool HasConfigDialog()
            {
                return false;
            }

            public virtual bool ShowConfigDialog(IWin32Window owner)
            {
                throw new NotImplementedException();
            }

            public override IFilter CreateFilter(IResizeableFilter sourceFilter)
            {
                throw new NotImplementedException();
            }

            public virtual void Destroy() { }

            public Preset()
            {
                Guid = Guid.NewGuid();
            }

            public static Preset Make<T>(string name = null)
                where T : IRenderChainUi, new()
            {
                var script = new T();
                return new SinglePreset() { Name = (name ?? script.Descriptor.Name), Script = script };
            }
        }

        public static class PresetHelper
        {
            public static Preset MakeNewPreset(this IRenderChainUi renderScript, string name = null)
            {
                return renderScript.CreateNew().ToPreset();
            }

            public static Preset ToPreset(this IRenderChainUi renderScript, string name = null)
            {
                return renderScript.GetChain() as Preset 
                    ?? new SinglePreset() { Name = (name ?? renderScript.Descriptor.Name), Script = renderScript };
            }
        }

        public class SinglePreset : Preset
        {
            public virtual IRenderChainUi Script { get; set; }

            public override bool HasConfigDialog()
            {
                return Script.HasConfigDialog();
            }

            public override bool ShowConfigDialog(IWin32Window owner)
            {
                return Script.ShowConfigDialog(owner);
            }

            public override void Destroy()
            {
                Script.Destroy();
            }

            public override string Description
            {
                get { return Script.Descriptor.Description;  }
            }

            public override IFilter CreateFilter(IResizeableFilter sourceFilter)
            {
                if (Script != null)
                    return Script.GetChain().CreateFilter(sourceFilter);
                else
                    return sourceFilter;
            }
        }

        public class MultiPreset : Preset
        {
            public IList<Preset> Options { get; set; }

            public MultiPreset()
                : base()
            {
                Options = new List<Preset>();
            }

            public override void Destroy()
            {
                foreach (var option in Options)
                    option.Destroy();
            }
        }

        public class PresetChain : MultiPreset
        {
            public override string Description
            {
                get { return string.Join(" ➔ ", Options.Select(x => x.Name)); }
            }

            public override IFilter CreateFilter(IResizeableFilter sourceFilter)
            {
                return Options.Aggregate(sourceFilter, (a, b) => a + b);
            }
        }

        public class PresetGroup : MultiPreset
        {
            #region Settings

            public int SelectedIndex { get; set; }

            public string Hotkey
            {
                get { return m_Hotkey; }
                set
                {
                    m_Hotkey = value ?? "";
                    RegisterHotkey();
                }
            }

            [YAXDontSerialize]
            public Preset SelectedOption { get { return Options.ElementAtOrDefault(SelectedIndex); } }

            #endregion

            public PresetGroup()
                : base()
            {
                SelectedIndex = -1;
            }

            public override string Description
            {
                get { return string.Join(", ", Options.Select(x => x.Name)); }
            }

            public override IFilter CreateFilter(IResizeableFilter sourceFilter)
            {
                if (SelectedOption != null)
                    return SelectedOption.CreateFilter(sourceFilter);
                else
                    return sourceFilter;
            }

            #region Hotkey Handling

            private string m_Hotkey;

            private void RegisterHotkey()
            {
                DynamicHotkeys.RegisterHotkey(Guid, Hotkey, IncrementSelection);
            }

            private void IncrementSelection()
            {
                if (Options.Count > 0)
                    SelectedIndex = (SelectedIndex + 1) % Options.Count;

                if (SelectedOption != null)
                    PlayerControl.ShowOsdText(Name + ": " + SelectedOption.Name);

                PlayerControl.SetRenderScript(PlayerControl.ActiveRenderScriptGuid);
            }

            #endregion
        }

        public class PresetChainDialog : PresetDialog
        {
            public PresetChainDialog()
                : base()
            {
                Text = "Script Chain";
            }
        }

        public class PresetGroupDialog : PresetDialog
        {
            public PresetGroupDialog()
                : base()
            {
                Text = "Script Group";
            }

            protected override void SaveSettings()
            {
                base.SaveSettings();
                (Settings as PresetGroup).SelectedIndex = SelectedIndex;
            }

            protected override void LoadSettings()
            {
                base.LoadSettings();
                SelectedIndex = (Settings as PresetGroup).SelectedIndex;
            }
        }

        public class ScriptChainScript : RenderChainUi<PresetChain, PresetChainDialog>
        {
            protected override string ConfigFileName
            {
                get { return "Mpdn.ScriptChain"; }
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
                        Guid = new Guid("3A462015-2D92-43AC-B559-396DACF896C3"),
                        Name = "Script Chain",
                        Description = Settings.Description != "" ? Settings.Description : "Applies all of a list of renderscripts"
                    };
                }
            }
        }

        public class ScriptGroupScript : RenderChainUi<PresetGroup, PresetGroupDialog>
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
                        Description = Settings.Description != "" ? Settings.Description : "Applies one of a list of renderscripts"
                    };
                }
            }
        }
    }
}
