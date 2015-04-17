using System;
using Mpdn.Config;

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
                MaxBitdepthSetter.Value = (Decimal)Settings.maxbitdepth;
                ThresholdSetter.Value = (Decimal)Settings.threshold;
                DetailSetter.Value = (Decimal)Settings.detaillevel;

                UpdateGui();
            }

            protected override void SaveSettings()
            {
                Settings.maxbitdepth = (int)MaxBitdepthSetter.Value;
                Settings.threshold = (float)ThresholdSetter.Value;
                Settings.detaillevel = (int)DetailSetter.Value;
            }

            private void ValueChanged(object sender, EventArgs e)
            {
                UpdateGui();
            }

            private void UpdateGui()
            {
            }
        }

        public class DebandConfigDialogBase : ScriptConfigDialog<Deband>
        {
        }
    }
}
