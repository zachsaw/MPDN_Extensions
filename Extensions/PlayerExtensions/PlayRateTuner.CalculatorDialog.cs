// This file is a part of MPDN Extensions.
// https://github.com/zachsaw/MPDN_Extensions
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
// 
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public
// License along with this library.
// 

using System.Globalization;
using System.Windows.Forms;

namespace Mpdn.Extensions.PlayerExtensions
{
    namespace GitHub
    {
        public partial class PlayrateTunerCalculatorDialog : Form
        {
            public PlayrateTunerCalculatorDialog()
            {
                InitializeComponent();
            }

            public void SetInputs(double displayHz, double videoHz, double refclk)
            {
                textBoxVsyncHz.Text = displayHz.ToString(CultureInfo.CurrentUICulture);
                textBoxFrameRateHz.Text = videoHz.ToString(CultureInfo.CurrentUICulture);
                textBoxRefClk.Text = refclk.ToString(CultureInfo.CurrentUICulture);
            }

            private void LinkCopyLinkClicked(object sender, LinkLabelLinkClickedEventArgs e)
            {
                var value = textBoxAnswer.Text.Trim();
                if (value.Length == 0)
                    return;

                Clipboard.SetText(value);
            }

            private void InputTextChanged(object sender, System.EventArgs e)
            {
                double displayHz = double.TryParse(textBoxVsyncHz.Text, NumberStyles.Any, CultureInfo.CurrentUICulture, out displayHz) ? displayHz : 0;
                double videoHz = double.TryParse(textBoxFrameRateHz.Text, NumberStyles.Any, CultureInfo.CurrentUICulture, out videoHz) ? videoHz : 0;
                double refClk = double.TryParse(textBoxRefClk.Text, NumberStyles.Any, CultureInfo.CurrentUICulture, out refClk) ? refClk : 0;

                Calculate(displayHz, videoHz, refClk);
            }

            private void Calculate(double displayHz, double videoHz, double refClk)
            {
                if (displayHz < 1e-8 || videoHz < 1e-8)
                {
                    textBoxAnswer.Text = 0.0.ToString(CultureInfo.CurrentUICulture);
                    return;
                }

                var result = ((displayHz / videoHz) * (1.0 - refClk)) * 100;
                textBoxAnswer.Text = result.ToString(CultureInfo.CurrentUICulture);
            }
        }
    }
}
