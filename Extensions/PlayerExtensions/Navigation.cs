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
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security;
using DirectShowLib;
using Mpdn.DirectShow;
using Mpdn.Extensions.Framework;

namespace Mpdn.Extensions.PlayerExtensions
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

        private readonly PlayerMenuItem[] m_MenuItems =
        {
            new PlayerMenuItem(initiallyDisabled: true),
            new PlayerMenuItem(initiallyDisabled: true),
            new PlayerMenuItem(initiallyDisabled: true),
            new PlayerMenuItem(initiallyDisabled: true),
            new PlayerMenuItem(initiallyDisabled: true),
            new PlayerMenuItem(initiallyDisabled: true),
            new PlayerMenuItem(initiallyDisabled: true),
            new PlayerMenuItem(initiallyDisabled: true),
            new PlayerMenuItem(initiallyDisabled: true),
            new PlayerMenuItem(initiallyDisabled: true)
        };

        private const int S_OK = 0;
        private static readonly Guid s_LavSourceFilterGuid = new Guid("B98D13E7-55DB-4385-A33D-09FD1BA26338");
        private long?[] m_KeyFrames;

        public override ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("79FFF20D-785B-497C-9716-066787F2A3AC"),
                    Name = "Navigation",
                    Description = "Adds shortcuts for rewinding / forwarding playback"
                };
            }
        }

        public override IList<Verb> Verbs
        {
            get
            {
                return new[]
                {
                    GetVerb("Forward (keyframe)", "Right", JumpKeyFrame(5), m_MenuItems[0]),
                    GetVerb("Backward (keyframe)", "Left", JumpKeyFrame(-5), m_MenuItems[1]),
                    GetVerb("Forward (1 frame)", "Ctrl+Right", StepFrame(), m_MenuItems[2]),
                    GetVerb("Backward (1 frame)", "Ctrl+Left", JumpFrame(-1), m_MenuItems[3]),
                    GetVerb("Forward (30 seconds)", "Ctrl+Shift+Right", Jump(30), m_MenuItems[4]),
                    GetVerb("Backward (30 seconds)", "Ctrl+Shift+Left", Jump(-30), m_MenuItems[5]),
                    GetVerb("Play next chapter", "Shift+Right", PlayChapter(true), m_MenuItems[6]),
                    GetVerb("Play previous chapter", "Shift+Left", PlayChapter(false), m_MenuItems[7]),
                    GetVerb("Play next file in folder", "Ctrl+PageDown", PlayFileInFolder(true), m_MenuItems[8]),
                    GetVerb("Play previous file in folder", "Ctrl+PageUp", PlayFileInFolder(false), m_MenuItems[9])
                };
            }
        }

        public override void Initialize()
        {
            base.Initialize();
            Player.StateChanged += PlayerStateChanged;
            Media.Loaded += OnMediaLoaded;
        }

        public override void Destroy()
        {
            Media.Loaded -= OnMediaLoaded;
            Player.StateChanged -= PlayerStateChanged;
            base.Destroy();
        }

        private static Filter VideoSourceFilter
        {
            get
            {
                return Player.Filters.FirstOrDefault(f => f.ClsId == s_LavSourceFilterGuid);
            }
        }

        private void OnMediaLoaded(object sender, EventArgs eventArgs)
        {
            m_KeyFrames = new long?[0];

            var filter = VideoSourceFilter;
            if (filter == null)
                return;

            ComThread.Do(() => SaveKeyFrameInfo(filter.Base));
        }

        private void PlayerStateChanged(object sender, PlayerStateEventArgs e)
        {
            foreach (var item in m_MenuItems)
            {
                item.Enabled = e.NewState != PlayerState.Closed;
            }
        }

        private static Verb GetVerb(string menuItemText, string shortCutString, Action action, PlayerMenuItem menuItem)
        {
            return new Verb(Category.Play, "Navigation", menuItemText, shortCutString, string.Empty, action, menuItem);
        }

        private void SaveKeyFrameInfo(IBaseFilter source)
        {
            var keyFrameInfo = (IKeyFrameInfo) source;
            int keyFrameCount;
            var hr = keyFrameInfo.GetKeyFrameCount(out keyFrameCount);
            if (hr != S_OK)
                return;

            var p = Marshal.AllocHGlobal(keyFrameCount*sizeof (long));
            try
            {
                if (keyFrameInfo.GetKeyFrames(TimeFormat.MediaTime, p, keyFrameCount) != S_OK)
                    return;

                var keyFrames = new long[keyFrameCount];
                Marshal.Copy(p, keyFrames, 0, keyFrameCount);
                m_KeyFrames = keyFrames
                    .Select(t => (long?) t/10) // convert to usec
                    .OrderBy(t => t) // make sure it's sorted in ascending order
                    .ToArray();
            }
            finally
            {
                Marshal.FreeHGlobal(p);
            }
        }

        private Action PlayChapter(bool next)
        {
            return () => SelectChapter(next);
        }

        private void SelectChapter(bool next)
        {
            if (Player.State == PlayerState.Closed)
                return;

            var chapters = Media.Chapters.OrderBy(chapter => chapter.Position);
            var pos = Media.Position;
            var nextChapter = next
                ? chapters.SkipWhile(chapter => chapter.Position < pos+1).FirstOrDefault()
                : chapters.TakeWhile(chapter => chapter.Position < Math.Max(pos-1000000, 0)).LastOrDefault();

            if (nextChapter != null)
            {
                Media.Seek(nextChapter.Position);
                Player.OsdText.Show(nextChapter.Name);
            }
        }

        private Action PlayFileInFolder(bool next)
        {
            return () => PlayFile(next);
        }

        private void PlayFile(bool next)
        {
            if (Player.State == PlayerState.Closed)
                return;

            var mediaPath = Media.FilePath;
            var mediaDir = GetDirectoryName(mediaPath);
            var mediaFiles = GetMediaFiles(mediaDir);
            var nextFile = next
                ? mediaFiles.SkipWhile(file => file != mediaPath).Skip(1).FirstOrDefault()
                : mediaFiles.TakeWhile(file => file != mediaPath).LastOrDefault();
            if (nextFile != null)
            {
                Media.Open(nextFile);
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

        private Action StepFrame()
        {
            return delegate
            {
                if (Player.State == PlayerState.Closed)
                    return;

                Media.Step();
            };
        }

        private Action JumpFrame(int frames)
        {
            return delegate
            {
                if (Player.State == PlayerState.Closed)
                    return;

                Media.Pause(false);
                var pos = Media.Position;
                var nextPos = pos + (long) Math.Round(frames*Media.VideoInfo.AvgTimePerFrame);
                nextPos = Math.Max(0, Math.Min(Media.Duration, nextPos));
                Media.Seek(nextPos);
            };
        }

        private Action JumpKeyFrame(float defaultTime)
        {
            return delegate
            {
                if (Player.State == PlayerState.Closed)
                    return;

                if (m_KeyFrames.Length == 0)
                {
                    Jump(defaultTime)();
                    return;
                }

                var forward = defaultTime >= 0;
                if (forward)
                {
                    var nextKeyFramePos = m_KeyFrames.FirstOrDefault(t => t > Media.Position);
                    if (nextKeyFramePos != null) Media.Seek(nextKeyFramePos.Value);
                    else Jump(defaultTime)();
                }
                else
                {
                    var prevKeyFramePos =
                        m_KeyFrames.LastOrDefault(
                            t => t < Media.Position - (Player.State == PlayerState.Playing ? 1000000 : 0));
                    if (prevKeyFramePos != null) Media.Seek(prevKeyFramePos.Value);
                    else Jump(defaultTime)();
                }
            };
        }

        private Action Jump(float time)
        {
            return delegate
            {
                if (Player.State == PlayerState.Closed)
                    return;

                var pos = Media.Position;
                var nextPos = pos + (long) Math.Round(time*1000*1000);
                nextPos = Math.Max(0, Math.Min(Media.Duration, nextPos));
                Media.Seek(nextPos);
            };
        }

        #region IKeyFrameInfo COM interface

        // See: https://github.com/Nevcairiel/LAVFilters/blob/master/developer_info/IKeyFrameInfo.h
        [ComImport, Guid("01A5BBD3-FE71-487C-A2EC-F585918A8724")]
        [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
        [SuppressUnmanagedCodeSecurity]
        private interface IKeyFrameInfo
        {
            [PreserveSig]
            int GetKeyFrameCount(out int nKFs); // returns S_FALSE when every frame is a keyframe

            [PreserveSig]
            int GetKeyFrames(ref Guid format, IntPtr pKFs, ref int nKFs);
        }

        #endregion
    }
}
