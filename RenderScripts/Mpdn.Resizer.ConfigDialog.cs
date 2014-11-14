using System;
using System.Windows.Forms;

namespace Mpdn.RenderScript
{
    namespace Mpdn.Resizer
    {
        public partial class ResizerConfigDialog : ScriptConfigDialog<Settings>
        {
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

            protected override void LoadSettings()
            {
                listBox.SelectedIndex = (int) Settings.Resizer;
            }

            protected override void SaveSettings()
            {
                Settings.Resizer = (ResizerOption) listBox.SelectedIndex;
            }

            private void ListBoxDoubleClick(object sender, EventArgs e)
            {
                DialogResult = DialogResult.OK;
            }
        }
    }
}