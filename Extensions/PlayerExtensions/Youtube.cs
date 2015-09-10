using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Windows.Forms;
using Mpdn.Extensions.Framework;

namespace Mpdn.Extensions.PlayerExtensions
{
    public class YoutubeExtension : PlayerExtension
    {
        private string m_DownloadHash;
        private Process m_Downloader;
        private readonly DirectoryInfo m_TempPath = new DirectoryInfo(Path.Combine(Path.GetTempPath(), @"MPDNTemp"));

        public override ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("062842A1-F991-4556-9706-4268F59F4880"),
                    Name = "Test",
                    Description = "Test",
                    Copyright = ""
                };
            }
        }

        public override void Initialize()
        {
            base.Initialize();
            PlayerControl.DragDrop += DragDrop;
        }

        public override void Destroy()
        {
            if (m_TempPath.Exists)
                foreach (FileInfo file in m_TempPath.GetFiles()) file.Delete();

            base.Destroy();
        }

        private void DragDrop(object sender, PlayerControlEventArgs<DragEventArgs> e)
        {
            var url = (string)e.InputArgs.Data.GetData(DataFormats.Text);
            if (url != null)// && url.Contains("youtu"))
            {
                LoadUrl(url);
                e.Handled = true;
            }
        }

        public override IList<Verb> Verbs
        {
            get { return new Verb[0]; }
        }

        private void PlayFile(string file)
        {
            Player.ActiveForm.Invoke((Action)(() => PlayerControl.OpenMedia(file)));
        }

        private void OnDownloaded(object sender, EventArgs e)
        {
            //if (m_Downloader.ExitCode != 0)
            //  throw new Exception(m_Downloader.StandardError.ReadToEnd());

            if (m_TempPath.Exists)
            {
                var file = m_TempPath.EnumerateFiles().FirstOrDefault(f => f.Name.Contains(m_DownloadHash));
                if (file != null)
                    PlayFile(file.FullName);
            }

            DisposeHelper.Dispose(ref m_Downloader);
        }

        private void OnWrite(object sender, DataReceivedEventArgs e)
        {
            Player.ActiveForm.Invoke((Action)(() => Player.OsdText.Show(e.Data)));
        }

        private void LoadUrl(string url)
        {
            m_DownloadHash = url.GetHashCode().ToString("X8");

            m_Downloader = new Process();
            m_Downloader.StartInfo = new ProcessStartInfo
            {
                FileName = "youtube-dl",
                Arguments = @"-f bestvideo[height<=?1080]+bestaudio/best " + "\"" + url + "\" " + " -o \"" + Path.Combine(m_TempPath.FullName, @"%(title)s_[" + m_DownloadHash + @"]_%(format_id)s.%(ext)s") + "\"",
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false
            };
            m_Downloader.EnableRaisingEvents = true;
            m_Downloader.Exited += OnDownloaded;
            m_Downloader.OutputDataReceived += OnWrite;

            m_Downloader.Start();
            m_Downloader.BeginOutputReadLine();
        }
    }
}