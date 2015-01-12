namespace Mpdn.PlayerExtensions
{
    namespace Example
    {
        public partial class DisplayChangerConfigDialog : DisplayChangerConfigBase
        {
            public DisplayChangerConfigDialog()
            {
                InitializeComponent();
            }

            protected override void LoadSettings()
            {
                checkBoxActivate.Checked = Settings.Activate;
            }

            protected override void SaveSettings()
            {
                Settings.Activate = checkBoxActivate.Checked;
            }
        }

        public class DisplayChangerConfigBase : ScriptConfigDialog<DisplayChangerSettings>
        {
        }
    }
}
