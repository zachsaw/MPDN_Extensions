using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Mpdn.PlayerExtensions.Example
{
    public class Navigation : PlayerExtension
    {
        private readonly string[] m_FileExtensions =
        {
            ".mkv", ".mp4", ".m4v", ".mp4v", ".3g2", ".3gp2", ".3gp", ".3gpp", 
            ".mov", ".m2ts", ".ts", ".asf", ".wma", ".wmv", ".wm", ".asx", 
            "*.wax", "*.wvx", "*.wmx", ".wpl", ".dvr-ms", ".avi", 
            ".mpg", ".mpeg", ".m1v", ".mp2", ".mp3", ".mpa", ".mpe", ".m3u", ".wav",
            ".mid", ".midi", ".rmi"
        };

        public override ExtensionDescriptor Descriptor
        {
            get
            {
                return new ExtensionDescriptor
                {
                    Guid = new Guid("79FFF20D-785B-497C-9716-066787F2A3AC"),
                    Name = "Navigation",
                    Description = "Adds shortcuts for rewinding / forwarding playback",
                    Copyright = "Copyright Example © 2014-2015. All rights reserved."
                };
            }
        }

        public override IList<Verb> Verbs
        {
            get
            {
                return new[]
                {
                    new Verb(Category.Play, "Navigation", "Forward (5 seconds)", "Right", string.Empty, Jump(5)),
                    new Verb(Category.Play, "Navigation", "Backward (5 seconds)", "Left", string.Empty, Jump(-5)),
                    new Verb(Category.Play, "Navigation", "Forward (1 frame)", "Ctrl+Right", string.Empty, JumpFrame(1)),
                    new Verb(Category.Play, "Navigation", "Backward (1 frame)", "Ctrl+Left", string.Empty, JumpFrame(-1)),
                    new Verb(Category.Play, "Navigation", "Forward (30 seconds)", "Ctrl+Shift+Right", string.Empty, Jump(30)),
                    new Verb(Category.Play, "Navigation", "Backward (30 seconds)", "Ctrl+Shift+Left", string.Empty, Jump(-30)),
                    new Verb(Category.Play, "Navigation", "Play Next file in folder", "Ctrl+PageDown", string.Empty, PlayNextFileInFolder),
                    new Verb(Category.Play, "Navigation", "Play Previous file in folder", "Ctrl+PageUp", string.Empty, PlayPreviousFileInFolder)
                };
            }
        }

        private void PlayNextFileInFolder()
        {
            if (PlayerControl.PlayerState == PlayerState.Closed)
                return;

            var mediaPath = PlayerControl.MediaFilePath;
            var mediaDir = Path.GetDirectoryName(mediaPath);
            if (mediaDir == null)
                return;

            var files = Directory.EnumerateFiles(mediaDir)
                .OrderBy(filename => filename).Where(file => m_FileExtensions.Contains(Path.GetExtension(file)));
            var nextFile = files.SkipWhile(file => file != mediaPath).Skip(1).FirstOrDefault();

            if (nextFile != null)
            {
                PlayerControl.OpenMedia(nextFile);
            }
        }

        private void PlayPreviousFileInFolder()
        {
            if (PlayerControl.PlayerState == PlayerState.Closed)
                return;

            var mediaPath = PlayerControl.MediaFilePath;
            var mediaDir = Path.GetDirectoryName(mediaPath);
            if (mediaDir == null)
                return;

            var files = Directory.EnumerateFiles(mediaDir)
                .OrderBy(filename => filename).Where(file => m_FileExtensions.Contains(Path.GetExtension(file)));
            var nextFile = files.TakeWhile(file => file != mediaPath).LastOrDefault();

            if (nextFile != null)
            {
                PlayerControl.OpenMedia(nextFile);
            }
        }

        private Action JumpFrame(int frames)
        {
            return delegate
            {
                if (PlayerControl.PlayerState == PlayerState.Closed)
                    return;

                var pos = PlayerControl.MediaPosition;
                var nextPos = pos + (long) Math.Round(frames*PlayerControl.VideoInfo.AvgTimePerFrame);
                nextPos = Math.Max(0, Math.Min(PlayerControl.MediaDuration, nextPos));
                PlayerControl.SeekMedia(nextPos);
            };
        }

        private Action Jump(float time)
        {
            return delegate
            {
                if (PlayerControl.PlayerState == PlayerState.Closed)
                    return;

                var pos = PlayerControl.MediaPosition;
                var nextPos = pos + (long) Math.Round(time*1000*1000);
                nextPos = Math.Max(0, Math.Min(PlayerControl.MediaDuration, nextPos));
                PlayerControl.SeekMedia(nextPos);
            };
        }
    }
}
