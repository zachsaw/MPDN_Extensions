using System;
using System.Linq;
using System.Windows.Forms;

namespace Mpdn.RenderScript
{
    namespace Shiandow.Chroma
    {
        public partial class ChromaScalerConfigDialog : Form
        {
            private Settings m_Settings;
            private bool m_SettingPreset;

            public ChromaScalerConfigDialog()
            {
                InitializeComponent();

                var descs = EnumHelpers.GetDescriptions<Presets>().Where(d => d != "Custom");
                foreach (var desc in descs)
                {
                    PresetBox.Items.Add(desc);
                }

                PresetBox.SelectedIndex = (int) Presets.Custom;
            }

            public void Setup(Settings settings)
            {
                m_Settings = settings;

                BSetter.Value = (Decimal) settings.B;
                CSetter.Value = (Decimal) settings.C;
                PresetBox.SelectedIndex = (int) settings.Preset;
            }

            private void ValueChanged(object sender, EventArgs e)
            {
                if (m_SettingPreset)
                    return;

                PresetBox.SelectedIndex = (int) Presets.Custom;
            }

            private void SelectedIndexChanged(object sender, EventArgs e)
            {
                var index = PresetBox.SelectedIndex;
                if (index < 0)
                    return;

                m_SettingPreset = true;
                BSetter.Value = (Decimal) ChromaScaler.BConst[index];
                CSetter.Value = (Decimal) ChromaScaler.CConst[index];
                m_SettingPreset = false;
            }

            private void DialogClosed(object sender, FormClosedEventArgs e)
            {
                if (DialogResult != DialogResult.OK)
                    return;

                m_Settings.B = (float) BSetter.Value;
                m_Settings.C = (float) CSetter.Value;
                m_Settings.Preset = (Presets) PresetBox.SelectedIndex;
            }
        }
    }
}
