using System;
using System.Windows.Forms;
using Mpdn.RenderScript.Config;

namespace Mpdn.RenderScript
{
    public class ScriptConfigDialog<TSettings> : Form
        where TSettings : class, new()
    {
        protected TSettings Settings { get; private set; }

        public virtual void Setup(TSettings settings)
        {
            Settings = settings;

            LoadSettings();
        }

        protected virtual void LoadSettings()
        {
            // This needs to be overriden
            throw new NotImplementedException();
        }

        protected virtual void SaveSettings()
        {
            // This needs to be overriden
            throw new NotImplementedException();
        }

        protected override void OnFormClosed(FormClosedEventArgs e)
        {
            base.OnFormClosed(e);

            if (DialogResult != DialogResult.OK)
                return;

            SaveSettings();
        }
    }

    public abstract class ConfigurableRenderChainUi<TChain, TDialog> : RenderChainUi<TChain>
        where TChain : class, IRenderChain, new()
        where TDialog : ScriptConfigDialog<TChain>, new()
    {
        protected Config ScriptConfig { get; private set; }
        protected override TChain Chain { get { return ScriptConfig.Config; } }

        protected abstract string ConfigFileName { get; }

        public override void Initialize()
        {
            ScriptConfig = new Config(ConfigFileName);
        }

        public override void Initialize(IRenderChain renderChain)
        {
            ScriptConfig = new Config(renderChain as TChain);
        }

        public override ScriptDescriptor Descriptor
        {
            get
            {
                return new ScriptDescriptor
                {
                    HasConfigDialog = true,
                    Guid = ScriptDescriptor.Guid,
                    Name = ScriptDescriptor.Name,
                    Description = ScriptDescriptor.Description,
                    Copyright = ScriptDescriptor.Copyright
                };
            }
        }

        public override bool ShowConfigDialog(IWin32Window owner)
        {
            using (var dialog = new TDialog())
            {
                dialog.Setup(ScriptConfig.Config);
                if (dialog.ShowDialog(owner) != DialogResult.OK)
                    return false;

                ScriptConfig.Save();
                return true;
            }
        }

        #region ScriptSettings Class

        public class Config : ScriptSettings<TChain>
        {
            private readonly string m_ConfigName;

            public Config(string configName)
            {
                m_ConfigName = configName;
                Load();
            }

            public Config(TChain Chain) : base(Chain) { }

            protected override string ScriptConfigFileName
            {
                get { return string.Format("{0}.config", m_ConfigName); }
            }
        }

        #endregion
    }
}