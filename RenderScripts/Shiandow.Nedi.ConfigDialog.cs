using System.Windows.Forms;

namespace Mpdn.RenderScript
{
    namespace Shiandow.Nedi
    {
        public partial class NediConfigDialog : Form
        {
            private Settings m_Settings;

            public NediConfigDialog()
            {
                InitializeComponent();
            }

            public void Setup(Settings settings)
            {
                m_Settings = settings;
                checkBoxAlwaysEnabled.Checked = m_Settings.AlwaysDoubleImage;
            }

            private void DialogClosed(object sender, FormClosedEventArgs e)
            {
                if (DialogResult != DialogResult.OK)
                    return;

                m_Settings.AlwaysDoubleImage = checkBoxAlwaysEnabled.Checked;
            }
        }
    }
}
