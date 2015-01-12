namespace Mpdn.PlayerExtensions
{
    namespace Example
    {
        public partial class MouseControlConfigDialog : MouseControlConfigBase
        {
            public MouseControlConfigDialog()
            {
                InitializeComponent();
            }

            protected override void LoadSettings()
            {
                checkBoxMouseWheelSeek.Checked = Settings.EnableMouseWheelSeek;
            }

            protected override void SaveSettings()
            {
                Settings.EnableMouseWheelSeek = checkBoxMouseWheelSeek.Checked;
            }
        }

        public class MouseControlConfigBase : ScriptConfigDialog<MouseControlSettings>
        {
        }
    }
}
