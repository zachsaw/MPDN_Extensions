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
                    GetVerb("Forward (5 seconds)", "Right", Jump(5)),
                    GetVerb("Backward (5 seconds)", "Left", Jump(-5)),
                    GetVerb("Forward (1 frame)", "Ctrl+Right", JumpFrame(1)),
                    GetVerb("Backward (1 frame)", "Ctrl+Left", JumpFrame(-1)),
                    GetVerb("Forward (30 seconds)", "Ctrl+Shift+Right", Jump(30)),
                    GetVerb("Backward (30 seconds)", "Ctrl+Shift+Left", Jump(-30)),
                    GetVerb("Play Next file in folder", "Ctrl+PageDown", PlayFileInFolder(true)),
                    GetVerb("Play Previous file in folder", "Ctrl+PageUp", PlayFileInFolder(false))
                };
            }
        }

        private static Verb GetVerb(string menuItemText, string shortCutString, Action action)
        {
            return new Verb(Category.Play, "Navigation", menuItemText, shortCutString, string.Empty, action);
        }

        private Action PlayFileInFolder(bool next)
        {
            return () => PlayFile(next);
        }

        private void PlayFile(bool next)
        {
            if (PlayerControl.PlayerState == PlayerState.Closed)
                return;

            var mediaPath = PlayerControl.MediaFilePath;
            var mediaDir = GetDirectoryName(mediaPath);
            var mediaFiles = GetMediaFiles(mediaDir);
            var nextFile = next
                ? mediaFiles.SkipWhile(file => file != mediaPath).Skip(1).FirstOrDefault()
                : mediaFiles.TakeWhile(file => file != mediaPath).LastOrDefault();
            if (nextFile != null)
            {
                PlayerControl.OpenMedia(nextFile);
            }
        }

        private static string GetDirectoryName(string path)
        {
            if (path == null)
            {
                throw new ArgumentNullException("path");
            }

            return Path.GetDirectoryName(path) ?? Path.GetPathRoot(path);
        }

        private IEnumerable<string> GetMediaFiles(string mediaDir)
        {
            var files = Directory.EnumerateFiles(mediaDir)
                .OrderBy(filename => filename).Where(file => m_FileExtensions.Contains(Path.GetExtension(file)));
            return files;
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