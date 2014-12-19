using System;
using System.Globalization;
using System.Windows.Forms;

namespace Mpdn.PlayerExtensions.ZachSaw
{
    public partial class GoToTimeForm : Form
    {
        public GoToTimeForm()
        {
            InitializeComponent();
        }

        public long Position
        {
            get
            {
                TimeSpan timespan;
                if (TimeSpan.TryParseExact(textBoxPos.Text, @"hh\:mm\:ss\.fff", CultureInfo.CurrentCulture, out timespan))
                    return (long) timespan.TotalMilliseconds;

                return -1;
            }
        }

        private void ButtonOkClick(object sender, EventArgs e)
        {
            if (Position < 0)
            {
                errorProvider.SetError(textBoxPos, "Invalid time");
                DialogResult = DialogResult.None;
            }
            else
            {
                errorProvider.SetError(textBoxPos, "");
            }
        }
    }
}
