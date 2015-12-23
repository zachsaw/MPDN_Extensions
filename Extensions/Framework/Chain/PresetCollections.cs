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

        protected PresetCollection()
        {
            Options = new List<Preset<T, TScript>>();
        }
    }

    public class ScriptChain<T, TScript> : PresetCollection<T, TScript>
        where TScript : class, IScript
    {
        public override T Process(T input)
        {
            return Options.Aggregate(input, (result, chain) => result + chain);
        }
    }

    public class ScriptGroup<T, TScript> : PresetCollection<T, TScript>
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

        public override T Process(T input)
        {
            return SelectedOption != null ? input + SelectedOption : input;
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
            {
                SelectedIndex = (SelectedIndex + 1) % Options.Count;
                
            }

            if (SelectedOption != null)
            {
                Player.OsdText.Show(Name + ": " + SelectedOption.Name);
                
                // Refresh everything until a better method can be found
                Extension.RefreshRenderScript();
            }
        }

        #endregion
    }
}
