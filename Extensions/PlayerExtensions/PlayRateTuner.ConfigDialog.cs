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

using System;
using System.Drawing;
using System.Globalization;
using System.Linq;
using System.Windows.Forms;
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.Framework.Config;
using Mpdn.Extensions.PlayerExtensions.GitHub;

namespace Mpdn.Extensions.PlayerExtensions
{
    public partial class PlayRateTunerConfigDialog : RateTunerConfigBase
    {
        public PlayRateTunerConfigDialog()
        {
            InitializeComponent();

            MinimumSize = Size;
            grid.CellBeginEdit += (sender, args) => { CancelButton = null; };
            grid.CellEndEdit += (sender, args) => { CancelButton = buttonCancel; };
        }

        protected override void LoadSettings()
        {
            checkBoxActivate.Checked = Settings.Activate;
            foreach (var t in Settings.Tunings)
            {
                grid.Rows.Add(t.Specifier, (t.Rate*100).ToString(CultureInfo.CurrentUICulture));
            }
        }

        protected override void SaveSettings()
        {
            Settings.Activate = checkBoxActivate.Checked;
            Settings.Tunings.Clear();
            foreach (DataGridViewRow r in grid.Rows)
            {
                if (r.IsNewRow)
                    continue;

                var specifier = Convert.ToString(r.Cells[0].FormattedValue).Trim();
                if (string.IsNullOrEmpty(specifier))
                    continue;

                double speed = double.TryParse(Convert.ToString(r.Cells[1].FormattedValue), NumberStyles.Any,
                    CultureInfo.CurrentUICulture, out speed)
                    ? speed
                    : 100;
                if (speed < 1e-8)
                    continue;

                Settings.Tunings.Add(new PlayRateTunerSettings.Tuning(specifier, speed/100));
            }
        }

        private void SelectionChanged(object sender, EventArgs e)
        {
            delButton.Enabled = grid.SelectedCells.Count > 0 && !grid.Rows[grid.SelectedCells[0].RowIndex].IsNewRow;
        }

        private void DelButtonClick(object sender, EventArgs e)
        {
            grid.Rows.RemoveAt(grid.SelectedCells[0].RowIndex);
            buttonOk.Enabled = !HasErrorText();
        }

        private void LinkClicked(object sender, LinkLabelLinkClickedEventArgs e)
        {
            var control = (Control) sender;
            toolTip.Show(string.Format(
                @"Video Type
    Format: {0}
    e.g.: {1}

Speed %
    e.g.
        150 to play 50% faster
        95.904 to do a PAL speedup correction
        104.271 to do a PAL speedup

You can also use this to match playback rate
with your display refresh rate (+refclock).

(Click here to dismiss)", VideoSpecifier.FormatHelp, VideoSpecifier.ExampleHelp), control,
                new Point(control.Width/2, control.Height));
        }

        private void CloseTips(object sender, EventArgs e)
        {
            toolTip.Hide(linkLabel);
        }

        private void GridKeyPress(object sender, KeyPressEventArgs e)
        {
            if (e.KeyChar == (char) Keys.Escape)
            {
                DialogResult = DialogResult.Cancel;
            }
        }

        private void GridCellValidating(object sender, DataGridViewCellValidatingEventArgs e)
        {
            var cell = ((DataGridView) sender).CurrentCell;
            var s = Convert.ToString(e.FormattedValue).Trim();
            switch (cell.ColumnIndex)
            {
                case 0:
                    if (ValidateSpecifier(cell, s))
                    {
                        cell.ErrorText = string.Empty;
                    }
                    break;
                case 1:
                    if (ValidateSpeed(cell, s))
                    {
                        cell.ErrorText = string.Empty;
                    }
                    break;
                default:
                    throw new IndexOutOfRangeException();
            }

            buttonOk.Enabled = !HasErrorText();
        }

        private bool HasErrorText()
        {
            return grid.Rows.Cast<DataGridViewRow>()
                .SelectMany(row => row.Cells.Cast<DataGridViewCell>())
                .Any(cell => cell.ErrorText.Length > 0);
        }

        private static bool ValidateSpeed(DataGridViewCell cell, string value)
        {
            if (string.IsNullOrWhiteSpace(value))
                return true;

            double v;
            if (double.TryParse(value, NumberStyles.Any, CultureInfo.CurrentUICulture, out v))
                return true;

            cell.ErrorText = "Error: Invalid number";
            return false;
        }

        private static bool ValidateSpecifier(DataGridViewCell cell, string value)
        {
            if (VideoSpecifier.IsValid(value))
                return true;

            cell.ErrorText = "Error: Invalid video type specifier - see usage tips";
            return false;
        }

        private void CalculatorClick(object sender, EventArgs e)
        {
            using (var form = new PlayrateTunerCalculatorDialog())
            {
                const int oneSecond = 1000000;
                var videoHz = Player.State == PlayerState.Closed
                    ? 0
                    : oneSecond/Stats.ActualSourceVideoIntervalUsec;
                var displayHz = oneSecond/Stats.DisplayRefreshIntervalUsec;
                var refclk = Player.State == PlayerState.Closed ||
                             Player.State == PlayerState.Stopped
                    ? 0
                    : Stats.RefClockDeviation;
                if (refclk > 10) // no data
                {
                    refclk = 0;
                }
                form.SetInputs(displayHz, videoHz, refclk);
                form.ShowDialog(this);
            }
        }

        private static IPlayerStats Stats
        {
            get { return Player.Stats.Details; }
        }
    }

    public class RateTunerConfigBase : ScriptConfigDialog<PlayRateTunerSettings>
    {
    }
}
