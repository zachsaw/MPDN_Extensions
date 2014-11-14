namespace Mpdn.RenderScript
{
    namespace Shiandow.Nedi
    {
        public partial class NediConfigDialog : ScriptConfigDialog<Settings>
        {
            public NediConfigDialog()
            {
                InitializeComponent();
            }

            protected override void LoadSettings()
            {
                checkBoxAlwaysEnabled.Checked = Settings.AlwaysDoubleImage;
            }

            protected override void SaveSettings()
            {
                Settings.AlwaysDoubleImage = checkBoxAlwaysEnabled.Checked;
            }
        }
    }
}
