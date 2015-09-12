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
            Player.DragDrop += DragDrop;
            Media.Loading += MediaLoading;
        }

        public override void Destroy()
        {
            if (m_TempPath.Exists)
                foreach (FileInfo file in m_TempPath.GetFiles()) file.Delete();

            Player.DragDrop -= DragDrop;
            Media.Loading -= MediaLoading;

            base.Destroy();
        }

        private void MediaLoading(object sender, MediaLoadingEventArgs e)
        {
            /* Doesn't work too well yet */
            if (IsValidUrl(e.Filename))
            {
                LoadUrl(e.Filename);
                Media.Close();
            }
        }

        private bool IsValidUrl(string uriName)
        {
            Uri uriResult;
            return Uri.TryCreate(uriName, UriKind.Absolute, out uriResult)
                && (   uriResult.Scheme == Uri.UriSchemeHttp
                    || uriResult.Scheme == Uri.UriSchemeHttps)
                && !Path.HasExtension(uriResult.AbsolutePath);
        }

        private void DragDrop(object sender, PlayerControlEventArgs<DragEventArgs> e)
        {
            var url = (string)e.InputArgs.Data.GetData(DataFormats.Text);
            if (IsValidUrl(url))
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
            var path = Path.Combine(m_TempPath.FullName, @"%(title)s_[" + m_DownloadHash + @"]_%(format_id)s.%(ext)s");

            m_Downloader = new Process();
            m_Downloader.StartInfo = new ProcessStartInfo
            {
                FileName = "youtube-dl",
                Arguments = String.Format("-f bestvideo[height<=?1080]+bestaudio/best \"{0}\" -o \"{1}\" --no-playlist", url, path),
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