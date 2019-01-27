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
            return Options
                .Aggregate(input, (temp, chain) => temp + chain)
                .Labeled(Description, 10, input);
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

        public ScriptGroup()
        {
            SelectedIndex = 0;
        }

        #endregion

        [YAXDontSerialize]
        public Preset<T, TScript> SelectedOption
        {
            get { return Options != null ? Options.ElementAtOrDefault(SelectedIndex) ?? Options.LastOrDefault() : null; }
        }

        public override T Process(T input)
        {
            return (SelectedOption != null 
                    ? input + SelectedOption 
                    : input)
                .Labeled(Description, 10, input);
        }

        #region Hotkey Handling

        private IDisposable m_HotkeyEntry;
        private string m_Hotkey;

        private void RegisterHotkey()
        {
            m_HotkeyEntry = HotkeyRegister.AddOrUpdateHotkey(Hotkey, IncrementSelection);
        }

        private void DeregisterHotkey()
        {
            DisposeHelper.Dispose(ref m_HotkeyEntry);
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
                
            // Refresh Scripts
            Extension.Refresh<TScript>();
        }

        #endregion
    }
}
