using System;
using System.Collections.Generic;
using Mpdn.PlayerExtensions.Config;
using Mpdn.RenderScript;
using YAXLib;

namespace Mpdn.PlayerExtensions.GitHub
{
    public interface IRenderScriptPreset
    {
        string Name { get; set; }
        Guid Guid { get; set; }
        IRenderScriptUi Script { get; set; }
    }

    public class RenderScriptPreset : IRenderScriptPreset
    {
        [YAXAttributeForClass]
        public string Name { get; set; }

        [YAXAttributeForClass]
        public Guid Guid { get; set; }

        public IRenderScriptUi Script { get; set; }

        public RenderScriptPreset()
        {
            Guid = Guid.NewGuid();
        }
    }

    public class PresetReference : IRenderScriptPreset
    {
        private IRenderScriptPreset Preset;

        [YAXAttributeForClass]
        public string Name
        {
            get { return Preset.Name; }
            set { Preset.Name = value; } 
        }

        [YAXAttributeForClass]
        public Guid Guid 
        {
            get { return Preset.Guid; }
            set 
            {
                Preset = RenderScriptExtension.LoadPreset(value) ?? Preset;
                Preset.Guid = value;
            }
        }

        public IRenderScriptUi Script
        {
            get { return Preset.Script; }
            set { Preset.Script = value; }
        }

        public PresetReference(IRenderScriptPreset preset)
        {
            Preset = preset;            
        }

        public PresetReference()
            : this(new RenderScriptPreset())
        {
        }
    }

    public class RenderScriptExtension : ConfigurablePlayerExtension<List<RenderScriptPreset>, RenderScriptExtensionDialog>
    {
        public static List<RenderScriptPreset> PresetList;

        protected static Action OnPresetChanged;
        private static RenderScriptPreset m_ActivePreset;
        public static RenderScriptPreset ActivePreset 
        {
            get { return m_ActivePreset; }
            set 
            { 
                m_ActivePreset = value;
                OnPresetChanged();
            }
        }
        public static Guid ScriptGuid;

        public static RenderScriptPreset LoadPreset(Guid guid)
        {
            return PresetList.Find(x => x.Guid == guid);
        }

        private void SelectRenderScript() 
        {
            if (ScriptGuid != Guid.Empty)
                PlayerControl.SetRenderScript(ScriptGuid);
        }

        public override IList<Verb> Verbs
        {
            get { return new Verb[] {}; }
        }

        public override void Initialize()
        {
            base.Initialize();

            RenderScriptExtension.PresetList = ScriptConfig.Config;
            RenderScriptExtension.OnPresetChanged = SelectRenderScript;
        }

        protected override string ConfigFileName
        {
            get { return "RenderScript"; }
        }

        protected override PlayerExtensionDescriptor ScriptDescriptor
        {
            get
            {
                return new PlayerExtensionDescriptor
                {
                    Guid = new Guid("26B49403-28D3-4C75-88C0-AB5372796CCC"),
                    Name = "RenderScript extension",
                    Description = "Extends renderscript funciontality with presets"
                };
            }
        }
    }
}