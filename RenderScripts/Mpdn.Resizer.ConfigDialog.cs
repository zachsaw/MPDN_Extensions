using System;
using System.Windows.Forms;

namespace Mpdn.RenderScript
{
    namespace Mpdn.Resizer
    {
        public partial class ResizerConfigDialog : Form
        {
            private Settings m_Settings;

            public ResizerConfigDialog()
            {
                InitializeComponent();

                var descs = EnumHelpers.GetDescriptions<ResizerOption>();
                foreach (var desc in descs)
                {
                    listBox.Items.Add(desc);
                }

                listBox.SelectedIndex = 0;
            }

            public void Setup(Settings settings)
            {
                m_Settings = settings;

                listBox.SelectedIndex = (int) settings.Resizer;
            }

            private void DialogClosed(object sender, FormClosedEventArgs e)
            {
                if (DialogResult != DialogResult.OK)
                    return;

                m_Settings.Resizer = (ResizerOption) listBox.SelectedIndex;
            }

            private void ListBoxDoubleClick(object sender, EventArgs e)
            {
                DialogResult = DialogResult.OK;
            }
        }
    }
}