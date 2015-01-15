using System;
using System.Linq;

namespace Mpdn.RenderScript
{
    namespace Shiandow.Deband
    {
        public partial class DebandConfigDialog : DebandConfigDialogBase
        {
            public DebandConfigDialog()
            {
                InitializeComponent();
            }

            protected override void LoadSettings()
            {
                PassesSetter.Value = (Decimal)Settings.maxbitdepth;
                StrengthSetter.Value = (Decimal)Settings.threshold;
                MarginSetter.Value = (Decimal)Settings.margin;

                UpdateText();
            }

            protected override void SaveSettings()
            {
                Settings.maxbitdepth = (int)PassesSetter.Value;
                Settings.threshold = (float)StrengthSetter.Value;
                Settings.margin = (float)MarginSetter.Value;
            }

            private void ValueChanged(object sender, EventArgs e)
            {
                UpdateText();
            }

            private void UpdateText() 
            {
                MaxErrorLabel.Text = String.Format("(maximum error {0:N2} bit)", (double)MarginSetter.Value * (Math.Sqrt(57) - 5) / 16);
            }

            private void AdvancedBox_CheckedChanged(object sender, EventArgs e)
            {
                panel1.Enabled = AdvancedBox.Checked;
            }
        }

        public class DebandConfigDialogBase : ScriptConfigDialog<Deband>
        {
        }
    }
}
