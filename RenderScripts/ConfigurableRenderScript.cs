using System;
using System.Windows.Forms;

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

    public abstract class ConfigurableRenderScript<TChain, TDialog> : RenderScript<TChain>
        where TChain : class, IRenderChain, new()
        where TDialog : ScriptConfigDialog<TChain>, new()
    {
        protected Config ScriptConfig { get; private set; }
        protected override TChain Chain { get { return ScriptConfig.Config; } }

        protected abstract string ConfigFileName { get; }

        public override void Initialize(int instanceId)
        {
            ScriptConfig = new Config(ConfigFileName, instanceId);
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
            private readonly int m_InstanceId;

            public Config(string configName, int instanceId)
                : base(false)
            {
                m_InstanceId = instanceId;
                m_ConfigName = configName;
                Load();
            }

            protected override string ScriptConfigFileName
            {
                get { return string.Format("{0}.{1}.config", m_ConfigName, m_InstanceId); }
            }
        }

        #endregion
    }
}