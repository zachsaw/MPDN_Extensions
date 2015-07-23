namespace Mpdn.Extensions.PlayerExtensions.OpenSubtitles
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

        public string Srt
        {
            get { return m_Downloader.FetchSubtitleText(this); }
        }

        public void Save()
        {
            m_Downloader.SaveSubtitleFile(this);
        }
    }
}