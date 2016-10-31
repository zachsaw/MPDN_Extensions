using System;
using System.Collections.Generic;
using System.Linq;
using YAXLib;

namespace Mpdn.Extensions.Framework.Chain
{
    public abstract class PresetCollection<T, TScript> : Chain<T>, INameable
        where TScript : class, IScript
    {
        #region Settings

        public List<Preset<T, TScript>> Options { get; set; }

        [YAXDontSerialize]
        public string Name { protected get; set; }

        #endregion

        protected virtual string Description
        {
            get
            {
                return String.IsNullOrEmpty(Name)
                    ? GetType().Name
                    : Name;
            }
        }

        protected PresetCollection()
        {
            Options = new List<Preset<T, TScript>>();
        }
    }

    public class ScriptChain<T, TScript> : PresetCollection<T, TScript>
        where T : ITagged
        where TScript : class, IScript
    {
        public override T Process(T input)
        {
            var result = Options.Aggregate(input, (temp, chain) => temp + chain);
            result.AddJunction(new StringTag(Description, 10), input);
            return result;
        }
    }

    public class ScriptGroup<T, TScript> : PresetCollection<T, TScript>
        where T : ITagged
        where TScript : class, IScript
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
        public Preset<T, TScript> SelectedOption
        {
            get { return Options != null ? Options.ElementAtOrDefault(SelectedIndex) ?? Options.LastOrDefault() : null; }
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

        public override T Process(T input)
        {
            var result = (SelectedOption != null ? input + SelectedOption : input);
            result.AddJunction(new StringTag(Description, 10), input);
            return result;
        }

        #region Hotkey Handling

        private readonly Guid m_HotkeyGuid;
        private string m_Hotkey;

        private void RegisterHotkey()
        {
            HotkeyRegister.RegisterHotkey(m_HotkeyGuid, Hotkey, IncrementSelection);
        }

        private void DeregisterHotkey()
        {
            HotkeyRegister.DeregisterHotkey(m_HotkeyGuid);
        }

        private void UpdateHotkey()
        {
            DeregisterHotkey();
            RegisterHotkey();
        }

        private void IncrementSelection()
        {
            if (Options.Count > 0)
                SelectedIndex = (SelectedIndex + 1) % Options.Count;

            if (SelectedOption == null)
                return;

            Player.OsdText.Show(Name + ": " + SelectedOption.Name);
                
            // Refresh everything (TODO: only refresh relevant scripts)
            Extension.RefreshRenderScript();
        }

        #endregion
    }
}
