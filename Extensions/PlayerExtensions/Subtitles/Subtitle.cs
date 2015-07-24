using System.IO;

namespace Mpdn.Extensions.PlayerExtensions.Subtitles
{
    public class Subtitle
    {
        public const string FILE_NAME_FORMAT = "{0}.{1}.srt";
        private readonly SubtitleDownloader m_Downloader;

        protected internal Subtitle(SubtitleDownloader downloader)
        {
            m_Downloader = downloader;
        }

        public string Lang { get; protected internal set; }
        public string Name { get; protected internal set; }
        public string Movie { get; protected internal set; }
        public int Id { get; protected internal set; }
        public string FilePath { get; protected internal set; }
        public string MediaFileName { get; protected internal set; }

        public string Srt
        {
            get { return m_Downloader.FetchSubtitleText(this); }
        }

        public void Save()
        {
            m_Downloader.SaveSubtitleFile(this);
        }

        public bool LoadSubtitle()
        {
            return File.Exists(FilePath) && SubtitleManager.LoadSubtitleFile(FilePath, Lang);
        }
    }
}