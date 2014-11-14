using System;
using System.Windows.Forms;

namespace Mpdn.RenderScript
{
    public abstract class ScriptConfigDialog<TSettings> : Form
        where TSettings : class, new()
    {
        protected TSettings Settings { get; private set; }

        public virtual void Setup(TSettings settings)
        {
            Settings = settings;

            LoadSettings();
        }

        protected abstract void LoadSettings();

        protected abstract void SaveSettings();

        protected override void OnFormClosed(FormClosedEventArgs e)
        {
            base.OnFormClosed(e);

            if (DialogResult != DialogResult.OK)
                return;

            SaveSettings();
        }
    }

    public class ConfigurableRenderScriptDescriptor
    {
        public Guid Guid = Guid.Empty;
        public string Name;
        public string Description;
        public string Copyright;
        public string ConfigFileName;
    }

    public abstract class ConfigurableRenderScript<TSettings, TDialog> : RenderScript
        where TSettings : class, new()
        where TDialog : ScriptConfigDialog<TSettings>, new()
    {
        protected Config Settings { get; private set; }

        protected abstract ConfigurableRenderScriptDescriptor ConfigScriptDescriptor { get; }

        public abstract IFilter CreateFilter(TSettings settings);

        protected virtual void Initialize(Config settings)
        {
        }

        #region Implementation

        private IFilter m_Filter;

        public override ScriptDescriptor Descriptor
        {
            get
            {
                return new ScriptDescriptor
                {
                    HasConfigDialog = true,
                    Guid = ConfigScriptDescriptor.Guid,
                    Name = ConfigScriptDescriptor.Name,
                    Description = ConfigScriptDescriptor.Description,
                    Copyright = ConfigScriptDescriptor.Copyright
                };
            }
        }

        public override void Setup(IRenderer renderer)
        {
            base.Setup(renderer);
            CreateFilter();
        }

        public override void Destroy()
        {
            Settings.Destroy();
        }

        public override void Initialize(int instanceId)
        {
            Settings = new Config(ConfigScriptDescriptor.ConfigFileName, instanceId);
            Initialize(Settings);
        }

        protected virtual void Initialize()
        {
            Settings = new Config();
            Initialize(Settings);
        }

        public override bool ShowConfigDialog(IWin32Window owner)
        {
            using (var dialog = new TDialog())
            {
                dialog.Setup(Settings.Config);
                if (dialog.ShowDialog(owner) != DialogResult.OK)
                    return false;

                Settings.Save();
                return true;
            }
        }

        public override IFilter GetFilter()
        {
            return m_Filter;
        }

        public void CreateFilter()
        {
            m_Filter = CreateFilter(Settings.Config);
            m_Filter.Initialize();
        }

        #endregion

        #region ScriptSettings Class

        public class Config : ScriptSettings<TSettings>
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

            public Config()
                : base(true)
            {
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