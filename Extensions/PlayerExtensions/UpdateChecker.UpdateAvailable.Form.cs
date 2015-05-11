using System;
using System.Diagnostics;
using System.Windows.Forms;
using Mpdn.PlayerExtensions.GitHub;

namespace OpenSubtitles.PlayerExtensions
{
    public partial class UpdateCheckerNewVersionForm : Form
    {
        private readonly UpdateCheckerSettings m_settings;
        public UpdateCheckerNewVersionForm(UpdateChecker.Version version, UpdateCheckerSettings settings)
        {
            InitializeComponent();
            m_settings = settings;
            Text += ": " + version;
            changelogBox.Text = version.Changelog;
            CancelButton = CloseButton;
        }

        private void forgetUpdate_Click(object sender, EventArgs e)
        {
            m_settings.ForgetVersion = true;
            Close();
        }

        private void downloadButton_Click(object sender, EventArgs e)
        {
            Process.Start(UpdateChecker.WebsiteUrl);
            Close();
        }
    }
}
