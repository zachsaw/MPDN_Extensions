using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Forms;
using Mpdn.RenderScript.Config;
using Mpdn.PlayerExtensions.GitHub;
using YAXLib;

namespace Mpdn.RenderScript
{
    namespace Mpdn.ScriptChain
    {
        public abstract class PresetRenderScript : IRenderScriptUi
        {
            protected abstract RenderScriptPreset Preset { get; }

            protected virtual IRenderScriptUi Script { get { return Preset.Script; } }

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

            public virtual ScriptDescriptor Descriptor
            {
                get 
                { 
                    var descriptor = Script.Descriptor;
                    descriptor.Guid = Preset.Guid;
                    return descriptor;
                }
            }
        }

        public class ActivePresetRenderScript : PresetRenderScript
        {
            private Guid m_Guid = new Guid("B1F3B882-3E8F-4A8C-B225-30C9ABD67DB1");

            protected override RenderScriptPreset Preset 
            {
                get { return PresetExtension.ActivePreset ?? new RenderScriptPreset() { Script = new ScriptChainScript() }; } 
            }

            public override void Initialize()
            {
                base.Initialize();

                PresetExtension.ScriptGuid = m_Guid;
            }

            public override ScriptDescriptor Descriptor
            {
                get
                {
                    var descriptor = base.Descriptor;
                    descriptor.Name = "Preset";
                    descriptor.Guid = m_Guid;
                    descriptor.Description = "Active Preset";
                    return descriptor;
                }
            }
        }
    }
}