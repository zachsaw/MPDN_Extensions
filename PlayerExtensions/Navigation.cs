using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Windows.Forms;

namespace Mpdn.PlayerExtensions.Example
{
    public class Navigation : PlayerExtension
    {
        private readonly string[] FileExtensions = {@".mkv",@".mp4",@".m4v",@".mp4v",@".3g2",@".3gp2",@".3gp",@".3gpp",@".mov",@".m2ts",@".ts",@".asf",@".wma",@".wmv",@".wm",@".asx,*.wax,*.wvx,*.wmx",@".wpl",@".dvr-ms",@".avi",@".mpg",@".mpeg",@".m1v",@".mp2",@".mp3",@".mpa",@".mpe",@".m3u",@".wav",@".mid",@".midi",@".rmi"};

        public override ExtensionDescriptor Descriptor
        {
            get
            {
                return new ExtensionDescriptor
                {
                    Guid = new Guid("79FFF20D-785B-497C-9716-066787F2A3AC"),
                    Name = "Navigation",
                    Description = "Adds shortcuts for rewinding / forwarding playback",
                    Copyright = "Copyright Example © 2014. All rights reserved."
                };
            }
        }

        public override IList<Verb> Verbs
        {
            get
            {
                return new[]
                {
                    new Verb(Category.Play, "Navigation", "Forward (5s)", "Right", string.Empty, Jump(5)),
                    new Verb(Category.Play, "Navigation", "Backward (5s)", "Left", string.Empty, Jump(-5)),
                    new Verb(Category.Play, "Navigation", "Forward (1 frame)", "Ctrl+Right", string.Empty, JumpFrame(1)),
                    new Verb(Category.Play, "Navigation", "Backward (1 frame)", "Ctrl+Left", string.Empty, JumpFrame(-1)),
                    new Verb(Category.Play, "Navigation", "Forward (30s)", "Ctrl+Shift+Right", string.Empty, Jump(30)),
                    new Verb(Category.Play, "Navigation", "Backward (30s)", "Ctrl+Shift+Left", string.Empty, Jump(-30)),
                    new Verb(Category.Play, "Navigation", "Play Next file in folder", "Ctrl+PageDown", string.Empty, PlayNextFileInFolder),
                    new Verb(Category.Play, "Navigation", "Play Next file in folder", "Ctrl+PageUp", string.Empty, PlayPreviousFileInFolder)
                };
            }
        }

        private void PlayNextFileInFolder() {
            var MediaPath = PlayerControl.MediaFilePath;
            var MediaDir = Path.GetDirectoryName(MediaPath);

            var files = Directory.EnumerateFiles(MediaDir)
                .Where(file => FileExtensions.Contains(Path.GetExtension(file)));
            var nextFile = files.SkipWhile(file => file != MediaPath).Skip(1).FirstOrDefault();

            if (nextFile != null)
                PlayerControl.OpenMedia(nextFile);
        }

        private void PlayPreviousFileInFolder()
        {
            var MediaPath = PlayerControl.MediaFilePath;
            var MediaDir = Path.GetDirectoryName(MediaPath);

            var files = Directory.EnumerateFiles(MediaDir)
                .Where(file => FileExtensions.Contains(Path.GetExtension(file)));
            var nextFile = files.TakeWhile(file => file != MediaPath).LastOrDefault();

            if (nextFile != null)
                PlayerControl.OpenMedia(nextFile);
        }

        private Action JumpFrame(int frames)
        {
            return delegate()
            {
                var pos = PlayerControl.MediaPosition;
                var nextPos = pos + (long)Math.Round(frames * PlayerControl.VideoInfo.AvgTimePerFrame);
                nextPos = Math.Max(0, Math.Min(PlayerControl.MediaDuration, nextPos));
                PlayerControl.SeekMedia(nextPos);
            };
        }

        private Action Jump(float time)
        {
            return delegate()
            {
                var pos = PlayerControl.MediaPosition;
                var nextPos = pos + (long)Math.Round(time * 1000 * 1000);
                nextPos = Math.Max(0, Math.Min(PlayerControl.MediaDuration, nextPos));
                PlayerControl.SeekMedia(nextPos);
            };
        }
    }
}