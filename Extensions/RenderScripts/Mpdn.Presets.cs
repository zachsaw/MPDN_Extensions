using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Forms;
using Mpdn.Config;
using Mpdn.PlayerExtensions.GitHub;
using YAXLib;

namespace Mpdn.RenderScript
{
    namespace Mpdn.Presets
    {
        public abstract class PresetRenderScriptBase : IRenderScriptUi
        {
            protected abstract RenderScriptPreset Preset { get; }

            protected virtual IRenderScriptUi Script { get { return Preset.Script ?? RenderScript.Empty; } }

            public virtual IRenderScript CreateRenderScript()
            {
                return Script.CreateRenderScript();
            }

            public virtual void Destroy()
            {
                Script.Destroy();
            }

            public virtual void Initialize() { }

            public virtual bool ShowConfigDialog(IWin32Window owner)
            {
                return Script.ShowConfigDialog(owner);
            }

            public virtual bool HasConfigDialog()
            {
                return Script.HasConfigDialog();
            }

            public virtual ExtensionUiDescriptor Descriptor
            {
                get 
                {
                    var descriptor = Script.Descriptor;
                    descriptor.Guid = Preset.Guid;
                    descriptor.Name = Preset.Name;
                    return descriptor;
                }
            }
        }

        public class PresetRenderScript : PresetRenderScriptBase
        {
            private RenderScriptPreset m_SavedPreset;
            public RenderScriptPreset SavedPreset  
            {
                get { return m_SavedPreset; }
                set
                {
                    m_SavedPreset = value;
                    m_SavedPreset = PresetExtension.LoadPreset(m_SavedPreset.Guid) ?? m_SavedPreset;
                }
            }

            public PresetRenderScript(RenderScriptPreset preset)
            {
                SavedPreset = preset;
            }

            protected override RenderScriptPreset Preset { get { return SavedPreset; } }
        }

        public class PresetRenderChain : PresetRenderScript, IRenderChainUi
        {
            public PresetRenderChain(RenderScriptPreset preset) : base(preset) 
            {
                if (!(Script is IRenderChainUi)) throw new ArgumentException("Not a preset for a RenderChain");
            }

            public RenderChain GetChain()
            {
                return (Script as IRenderChainUi).GetChain();
            }
        }

        public class ActivePresetRenderScript : PresetRenderScriptBase
        {
            private Guid m_Guid = new Guid("B1F3B882-3E8F-4A8C-B225-30C9ABD67DB1");

            protected override RenderScriptPreset Preset 
            {
                get { return PresetExtension.ActivePreset ?? new RenderScriptPreset() { Script = RenderScript.Empty }; } 
            }

            public override void Initialize()
            {
                base.Initialize();

                PresetExtension.ScriptGuid = m_Guid;
            }

            public override ExtensionUiDescriptor Descriptor
            {
                get
                {
                    var descriptor = base.Descriptor;
                    descriptor.Name = "Preset";
                    descriptor.Guid = m_Guid;
                    descriptor.Description = "Active Preset: " + Preset.Name;
                    return descriptor;
                }
            }
        }
    }
}