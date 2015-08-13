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
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Windows.Forms;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using DirectShowLib;
using Mpdn.AudioScript;
using Mpdn.Config;
using Mpdn.DirectShow;
using Mpdn.RenderScript;
using Control = System.Windows.Forms.Control;
using WaveFormatExtensible = DirectShowLib.WaveFormatExtensible;

namespace Mpdn.Extensions.Framework
{
    public static class DisposeHelper
    {
        public static void Dispose(object obj)
        {
            var disposable = obj as IDisposable;
            if (disposable != null)
            {
                disposable.Dispose();
            }
        }

        public static void Dispose<T>(ref T obj) where T : class, IDisposable
        {
            if (obj != null)
            {
                obj.Dispose();
                obj = default(T);
            }
        }
    }

    public static class PathHelper
    {
        public static string GetDirectoryName(string path)
        {
            if (path == null)
            {
                throw new ArgumentNullException("path");
            }

            return Path.GetDirectoryName(path) ?? Path.GetPathRoot(path);
        }

        public static string GetExtension(string path)
        {
            if (path == null)
            {
                throw new ArgumentNullException("path");
            }

            var result = Path.GetExtension(path);
            if (result == null)
                throw new ArgumentException();

            return result;
        }

        public static string ExtensionsPath
        {
            get
            {
                return Path.Combine(GetDirectoryName(Assembly.GetAssembly(typeof(IPlayerExtension)).Location),
                    "Extensions");
            }
        }
    }

    public static class EnumHelpers
    {
        public static string ToDescription(this Enum en)
        {
            var type = en.GetType();
            var enumString = en.ToString();

            var memInfo = type.GetMember(enumString);

            if (memInfo.Length > 0)
            {
                var attrs = memInfo[0].GetCustomAttributes(typeof(DescriptionAttribute), false);

                if (attrs.Length > 0)
                {
                    return ((DescriptionAttribute)attrs[0]).Description;
                }
            }

            return enumString;
        }

        public static string[] GetDescriptions<T>()
        {
            return Enum.GetValues(typeof(T)).Cast<Enum>().Select(val => val.ToDescription()).ToArray();
        }
    }

    public static class RendererHelpers
    {
        public static TextureFormat GetTextureFormat(this RenderQuality quality)
        {
            switch (quality)
            {
                case RenderQuality.MaxPerformance:
                    return TextureFormat.Unorm8;
                case RenderQuality.Performance:
                    return TextureFormat.Float16;
                case RenderQuality.Quality:
                    return TextureFormat.Unorm16;
                case RenderQuality.MaxQuality:
                    return TextureFormat.Float32;
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        public static bool PerformanceMode(this RenderQuality quality)
        {
            switch (quality)
            {
                case RenderQuality.MaxPerformance:
                case RenderQuality.Performance:
                    return true;
                case RenderQuality.Quality:
                case RenderQuality.MaxQuality:
                    return false;
                default:
                    throw new ArgumentOutOfRangeException("quality");
            }
        }

        public static bool QualityMode(this RenderQuality quality)
        {
            return !PerformanceMode(quality);
        }

        public static bool IsFullRange(this YuvColorimetric colorimetric)
        {
            return !IsLimitedRange(colorimetric);
        }

        public static bool IsLimitedRange(this YuvColorimetric colorimetric)
        {
            switch (colorimetric)
            {
                case YuvColorimetric.FullRangePc601:
                case YuvColorimetric.FullRangePc709:
                case YuvColorimetric.FullRangePc2020:
                    return false;
                case YuvColorimetric.ItuBt601:
                case YuvColorimetric.ItuBt709:
                case YuvColorimetric.ItuBt2020:
                    return true;
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        public static bool IsYuv(this FrameBufferInputFormat format)
        {
            return !IsRgb(format);
        }

        public static bool IsRgb(this FrameBufferInputFormat format)
        {
            switch (format)
            {
                case FrameBufferInputFormat.Rgb24:
                case FrameBufferInputFormat.Rgb32:
                case FrameBufferInputFormat.Rgb48:
                    return true;
            }

            return false;
        }

        public static int GetBitDepth(this FrameBufferInputFormat format)
        {
            switch (format)
            {
                case FrameBufferInputFormat.Nv12:
                case FrameBufferInputFormat.Yv12:
                case FrameBufferInputFormat.Yuy2:
                case FrameBufferInputFormat.Uyvy:
                case FrameBufferInputFormat.Yv24:
                case FrameBufferInputFormat.Ayuv:
                case FrameBufferInputFormat.Rgb24:
                case FrameBufferInputFormat.Rgb32:
                    return 8;
                case FrameBufferInputFormat.P010:
                case FrameBufferInputFormat.P210:
                case FrameBufferInputFormat.Y410:
                    return 10;
                case FrameBufferInputFormat.P016:
                case FrameBufferInputFormat.P216:
                case FrameBufferInputFormat.Y416:
                case FrameBufferInputFormat.Rgb48:
                    return 16;
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        public static int ToInt(this ScalerTaps scalerTaps)
        {
            switch (scalerTaps)
            {
                case ScalerTaps.Two:
                    return 2;
                case ScalerTaps.Four:
                    return 4;
                case ScalerTaps.Six:
                    return 6;
                case ScalerTaps.Eight:
                    return 8;
                case ScalerTaps.Twelve:
                    return 12;
                case ScalerTaps.Sixteen:
                    return 16;
                default:
                    throw new ArgumentOutOfRangeException("scalerTaps");
            }
        }
        
        public static float[] GetYuvConsts(this YuvColorimetric colorimetric) 
        {
            switch (colorimetric)
            {
                case YuvColorimetric.FullRangePc601:
                case YuvColorimetric.ItuBt601:
                    return new[] {0.114f, 0.299f};
                case YuvColorimetric.FullRangePc709:
                case YuvColorimetric.ItuBt709:
                    return new[] {0.0722f, 0.2126f};
                case YuvColorimetric.FullRangePc2020:
                case YuvColorimetric.ItuBt2020:
                    return new[] {0.0593f, 0.2627f};
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }
    }

    public enum AudioSampleFormat
    {
        Unknown,
        Float,
        Double,
        Pcm8,
        Pcm16,
        Pcm24,
        Pcm32
    }

    public static class AudioHelpers
    {
        private const int S_OK = 0;

        private const short WAVE_FORMAT_PCM = 1;
        private const short WAVE_FORMAT_IEEE_FLOAT = 3;
        private const short WAVE_FORMAT_EXTENSIBLE = unchecked((short) 0xFFFE);

        [DllImport("kernel32.dll", EntryPoint = "CopyMemory", SetLastError = false)]
        public static extern void CopyMemory(IntPtr dest, IntPtr src, int count);

        public static void CopySample(IMediaSample src, IMediaSample dest)
        {
            var sourceSize = src.GetActualDataLength();
            var destSize = dest.GetSize();

            Debug.Assert(destSize >= sourceSize);

            IntPtr sourceBuffer;
            src.GetPointer(out sourceBuffer);

            IntPtr destBuffer;
            dest.GetPointer(out destBuffer);

            CopyMemory(destBuffer, sourceBuffer, sourceSize);

            // Copy the sample times
            long start, end;

            if (src.GetTime(out start, out end) == S_OK)
            {
                dest.SetTime(start, end);
            }

            if (src.GetMediaTime(out start, out end) == S_OK)
            {
                dest.SetMediaTime(start, end);
            }

            // Copy the media type
            AMMediaType mediaType;
            src.GetMediaType(out mediaType);
            dest.SetMediaType(mediaType);
            DsUtils.FreeAMMediaType(mediaType);

            dest.SetSyncPoint(src.IsSyncPoint() == S_OK);
            dest.SetPreroll(src.IsPreroll() == S_OK);
            dest.SetDiscontinuity(src.IsDiscontinuity() == S_OK);

            // Copy the actual data length
            dest.SetActualDataLength(sourceSize);
        }

        public static AudioSampleFormat SampleFormat(this WaveFormatExtensible format)
        {
            return GetSampleFormat(format);
        }

        public static AudioSampleFormat GetSampleFormat(WaveFormatExtensible format)
        {
            if (format.nSamplesPerSec == 0)
                return AudioSampleFormat.Unknown;

            switch (format.wFormatTag)
            {
                case WAVE_FORMAT_IEEE_FLOAT:
                    switch (format.wBitsPerSample)
                    {
                        case 32: return AudioSampleFormat.Float;
                        case 64: return AudioSampleFormat.Double;
                    }
                    break;
                case WAVE_FORMAT_PCM:
                    switch (format.wBitsPerSample)
                    {
                        case 8:  return AudioSampleFormat.Pcm8;
                        case 16: return AudioSampleFormat.Pcm16;
                        case 24: return AudioSampleFormat.Pcm24;
                        case 32: return AudioSampleFormat.Pcm32;
                    }
                    break;
                case WAVE_FORMAT_EXTENSIBLE:
                    if (format.SubFormat == MediaSubType.IEEE_FLOAT)
                    {
                        switch (format.wBitsPerSample)
                        {
                            case 32: return AudioSampleFormat.Float;
                            case 64: return AudioSampleFormat.Double;
                        }
                    }
                    else if (format.SubFormat == MediaSubType.PCM)
                    {
                        switch (format.wBitsPerSample)
                        {
                            case 8:  return AudioSampleFormat.Pcm8;
                            case 16: return AudioSampleFormat.Pcm16;
                            case 24: return AudioSampleFormat.Pcm24;
                            case 32: return AudioSampleFormat.Pcm32;
                        }
                    }
                    break;
            }

            return AudioSampleFormat.Unknown;
        }

        public static bool IsBitStreaming(this WaveFormatExtensible format)
        {
            return IsBitstreaming(format);
        }

        public static bool IsBitstreaming(WaveFormatExtensible format)
        {
            return GetSampleFormat(format) == AudioSampleFormat.Unknown;
        }
    }

    public static class AudioKernels
    {
        [Cudafy]
        public static void GetSamplesFloat(GThread thread, float[] samples, float[,] output)
        {
            var channels = output.GetLength(0);
            var sampleCount = output.GetLength(1);

            int tid = thread.blockIdx.x;
            while (tid < sampleCount)
            {
                for (int i = 0; i < channels; i++)
                {
                    output[i, tid] = samples[(tid*channels) + i];
                }
                tid += thread.gridDim.x;
            }
        }

        [Cudafy]
        public static void PutSamplesFloat(GThread thread, float[,] samples, float[] output)
        {
            var channels = samples.GetLength(0);
            var sampleCount = samples.GetLength(1);

            int tid = thread.blockIdx.x;
            while (tid < sampleCount)
            {
                for (int i = 0; i < channels; i++)
                {
                    output[(tid*channels) + i] = samples[i, tid];
                }
                tid += thread.gridDim.x;
            }
        }

        private static CudafyModule s_KernelModule;

        public static CudafyModule KernelModule
        {
            get
            {
                return s_KernelModule ??
                       (s_KernelModule = CudafyTranslator.Cudafy(eArchitecture.OpenCL, typeof (AudioKernels)));
            }
        }
    }

    public static class StringHelpers
    {
        public static string SubstringIdx(this string self, int startIndex, int endIndex)
        {
            var length = endIndex - startIndex;
            if (length < 0)
            {
                length = 0;
            }
            return self.Substring(startIndex, length);
        }
    }

    public static class EventHelpers
    {
        public static void Handle<T>(this T self, Action<T> action) where T : class
        {
            if (self == null)
                return;

            action(self);
        }
    }

    public static class Gui
    {
        public static Control VideoBox
        {
            get { return PlayerControl.VideoPanel; }
        }

        public static Icon Icon
        {
            get { return PlayerControl.ApplicationIcon; }
        }

        public static int FullScreenSeekBarHeight
        {
            get { return PlayerControl.FullScreenSeekBarHeight; }
        }
    }

    public static class ComThread
    {
        public static void Do(Action action)
        {
            PlayerControl.ComInvoke(action);
        }

        public static void DoAsync(Action action)
        {
            PlayerControl.ComInvokeAsync(action);
        }
    }

    public static class Extension
    {
        public static int InterfaceVersion
        {
            get { return 3; }
        }

        public static IList<Assembly> Assemblies
        {
            get { return PlayerControl.ExtensionAssemblies; }
        }

        public static IList<IRenderScriptUi> RenderScripts
        {
            get { return PlayerControl.RenderScripts; }
        }

        public static IList<IAudioScript> AudioScripts
        {
            get { return PlayerControl.AudioScripts; }
        }

        public static IList<IPlayerExtension> PlayerExtensions
        {
            get { return PlayerControl.PlayerExtensions; }
        }

        public static IAudioScript AudioScript
        {
            get { return PlayerControl.ActiveAudioScript; }
        }

        public static IRenderScript RenderScript
        {
            get { return PlayerControl.ActiveRenderScript; }
        }

        public static Guid RenderScriptGuid
        {
            get { return PlayerControl.ActiveRenderScriptGuid; }
            set { PlayerControl.SetRenderScript(value); }
        }

        public static void SetRenderScript(Guid renderScriptGuid)
        {
            PlayerControl.SetRenderScript(renderScriptGuid);
        }
    }

    public static class Player
    {
        public static Form ActiveForm
        {
            get { return PlayerControl.Form; }
        }

        public static class Stats
        {
            public static IPlayerStats Details
            {
                get { return PlayerControl.PlayerStats; }
            }

            public static void Show(bool show)
            {
                PlayerControl.ShowStats(show);
            }

            public static void Reset()
            {
                PlayerControl.ResetStats();
            }
        }

        public static PlayerState State
        {
            get { return PlayerControl.PlayerState; }
        }

        public static int Volume
        {
            get { return PlayerControl.Volume; }
            set { PlayerControl.Volume = value; }
        }

        public static bool Mute
        {
            get { return PlayerControl.Mute; }
            set { PlayerControl.Mute = value; }
        }

        public static class Playback
        {
            public static event EventHandler RateChanged
            {
                add { PlayerControl.PlaybackRateChanged += value; }
                remove { PlayerControl.PlaybackRateChanged -= value; }
            }


            public static event EventHandler Completed
            {
                add { PlayerControl.PlaybackCompleted += value; }
                remove { PlayerControl.PlaybackCompleted -= value; }
            }

            public static double Rate
            {
                get { return PlayerControl.PlaybackRate; }
                set { PlayerControl.PlaybackRate = value; }
            }

            public static double BaseRate
            {
                get { return PlayerControl.PlaybackBaseRate; }
                set { PlayerControl.PlaybackBaseRate = value; }
            }
        }

        public static class Config
        {
            public static event EventHandler Changed
            {
                add { PlayerControl.SettingsChanged += value; }
                remove { PlayerControl.SettingsChanged -= value; }
            }

            public static string Path
            {
                get { return PlayerControl.ConfigRootPath; }
            }

            public static Configuration Settings
            {
                get { return PlayerControl.PlayerSettings; }
            }

            public static void Refresh()
            {
                PlayerControl.RefreshSettings();
            }

            public static bool Commit()
            {
                return PlayerControl.CommitSettings();
            }
        }

        public static class FullScreenMode
        {
            public static event EventHandler Entering
            {
                add { PlayerControl.EnteringFullScreenMode += value; }
                remove { PlayerControl.EnteringFullScreenMode -= value; }
            }

            public static event EventHandler Entered
            {
                add { PlayerControl.EnteredFullScreenMode += value; }
                remove { PlayerControl.EnteredFullScreenMode -= value; }
            }

            public static event EventHandler Exiting
            {
                add { PlayerControl.ExitingFullScreenMode += value; }
                remove { PlayerControl.ExitingFullScreenMode -= value; }
            }

            public static event EventHandler Exited
            {
                add { PlayerControl.ExitedFullScreenMode += value; }
                remove { PlayerControl.ExitedFullScreenMode -= value; }
            }

            public static bool Active
            {
                get { return PlayerControl.InFullScreenMode; }
                set
                {
                    if (value == PlayerControl.InFullScreenMode)
                        return;

                    if (value)
                    {
                        PlayerControl.GoFullScreen();
                    }
                    else
                    {
                        PlayerControl.GoWindowed();
                    }
                }
            }
        }

        public static event EventHandler Loaded
        {
            add { PlayerControl.PlayerLoaded += value; }
            remove { PlayerControl.PlayerLoaded -= value; }
        }

        public static event EventHandler Closed
        {
            add { PlayerControl.FormClosed += value; }
            remove { PlayerControl.FormClosed -= value; }
        }

        public static event EventHandler PaintOverlay
        {
            add { PlayerControl.PaintOverlay += value; }
            remove { PlayerControl.PaintOverlay -= value; }
        }

        public static event EventHandler<PlayerStateEventArgs> StateChanged
        {
            add { PlayerControl.PlayerStateChanged += value; }
            remove { PlayerControl.PlayerStateChanged -= value; }
        }

        public static event EventHandler VolumeChanged
        {
            add { PlayerControl.VolumeChanged += value; }
            remove { PlayerControl.VolumeChanged -= value; }
        }

        public static event EventHandler<PlayerControlEventArgs<KeyEventArgs>> KeyDown
        {
            add { PlayerControl.KeyDown += value; }
            remove { PlayerControl.KeyDown -= value; }
        }

        public static event EventHandler<PlayerControlEventArgs<MouseEventArgs>> MouseWheel
        {
            add { PlayerControl.MouseWheel += value; }
            remove { PlayerControl.MouseWheel -= value; }
        }

        public static event EventHandler<PlayerControlEventArgs<MouseEventArgs>> MouseMove
        {
            add { PlayerControl.MouseMove += value; }
            remove { PlayerControl.MouseMove -= value; }
        }

        public static event EventHandler<PlayerControlEventArgs<MouseEventArgs>> MouseClick
        {
            add { PlayerControl.MouseClick += value; }
            remove { PlayerControl.MouseClick -= value; }
        }

        public static event EventHandler<PlayerControlEventArgs<MouseEventArgs>> MouseDoubleClick
        {
            add { PlayerControl.MouseDoubleClick += value; }
            remove { PlayerControl.MouseDoubleClick -= value; }
        }

        public static event EventHandler<PlayerControlEventArgs<DragEventArgs>> DragDrop
        {
            add { PlayerControl.DragDrop += value; }
            remove { PlayerControl.DragDrop -= value; }
        }

        public static event EventHandler<PlayerControlEventArgs<DragEventArgs>> DragEnter
        {
            add { PlayerControl.DragEnter += value; }
            remove { PlayerControl.DragEnter -= value; }
        }

        public static event EventHandler<CommandLineFileOpenEventArgs> CommandLineFileOpen
        {
            add { PlayerControl.CommandLineFileOpen += value; }
            remove { PlayerControl.CommandLineFileOpen -= value; }
        }

        public static class Window
        {
            public static void FitAspectRatio()
            {
                PlayerControl.FitAspectRatio();
            }

            public static void FitPercentage(int percent)
            {
                PlayerControl.FitPercentage(percent);
            }

            public static void ResetSize()
            {
                PlayerControl.ResetWindowSize();
            }
        }

        public static void HandleException(Exception exception)
        {
            PlayerControl.HandleException(exception);
        }

        public static class OsdText
        {
            public static void Show(string text, int durationMs = 1000)
            {
                PlayerControl.ShowOsdText(text, durationMs);
            }

            public static void Hide()
            {
                PlayerControl.HideOsdText();
            }
        }

        public static IText CreateText(string font, int size, TextFontStyle style)
        {
            return PlayerControl.CreateText(font, size, style);
        }

        public static void ClearScreen()
        {
            PlayerControl.ClearScreen();
        }

        public static void ShowOptionsDialog()
        {
            PlayerControl.ShowOptionsDialog();
        }

        public static class Filters
        {
            public static IList<Filter> Audio
            {
                get { return PlayerControl.AudioFilters; }
            }

            public static IList<Filter> Video
            {
                get { return PlayerControl.VideoFilters; }
            }
        }
    }

    public static class Media
    {
        public static string FilePath
        {
            get { return PlayerControl.MediaFilePath; }
        }

        public static long Duration
        {
            get { return PlayerControl.MediaDuration; }
        }

        public static long Position
        {
            get { return PlayerControl.MediaPosition; }
        }

        public static IList<Chapter> Chapters
        {
            get { return PlayerControl.Chapters; }
        }

        public static event EventHandler AudioTrackChanged
        {
            add { PlayerControl.AudioTrackChanged += value; }
            remove { PlayerControl.AudioTrackChanged -= value; }
        }

        public static event EventHandler VideoTrackChanged
        {
            add { PlayerControl.VideoTrackChanged += value; }
            remove { PlayerControl.VideoTrackChanged -= value; }
        }

        public static event EventHandler SubtitleTrackChanged
        {
            add { PlayerControl.SubtitleTrackChanged += value; }
            remove { PlayerControl.SubtitleTrackChanged -= value; }
        }

        public static MediaTrack AudioTrack
        {
            get { return PlayerControl.ActiveAudioTrack; }
            set { SelectAudioTrack(value); }
        }

        public static MediaTrack VideoTrack
        {
            get { return PlayerControl.ActiveVideoTrack; }
            set { SelectVideoTrack(value); }
        }

        public static MediaTrack SubtitleTrack
        {
            get { return PlayerControl.ActiveSubtitleTrack; }
            set { SelectSubtitleTrack(value); }
        }

        public static void SelectAudioTrack(MediaTrack track, bool showOsd = true)
        {
            PlayerControl.SelectAudioTrack(track, showOsd);
        }

        public static void SelectVideoTrack(MediaTrack track, bool showOsd = true)
        {
            PlayerControl.SelectVideoTrack(track, showOsd);
        }

        public static void SelectSubtitleTrack(MediaTrack track, bool showOsd = true)
        {
            PlayerControl.SelectSubtitleTrack(track, showOsd);
        }

        public static IList<MediaTrack> AudioTracks
        {
            get { return PlayerControl.AudioTracks; }
        }

        public static IList<MediaTrack> VideoTracks
        {
            get { return PlayerControl.VideoTracks; }
        }

        public static IList<MediaTrack> SubtitleTracks
        {
            get { return PlayerControl.SubtitleTracks; }
        }

        public static VideoInfo VideoInfo
        {
            get { return PlayerControl.VideoInfo; }
        }

        public static AMMediaType VideoMediaType
        {
            get { return PlayerControl.VideoMediaType; }
        }

        public static event EventHandler Loaded
        {
            add { PlayerControl.MediaLoaded += value; }
            remove { PlayerControl.MediaLoaded -= value; }
        }

        public static event EventHandler<MediaLoadingEventArgs> Loading
        {
            add { PlayerControl.MediaLoading += value; }
            remove { PlayerControl.MediaLoading -= value; }
        }

        public static void Open(string filename, bool play = true, bool showOsd = true)
        {
            PlayerControl.OpenMedia(filename, play, showOsd);
        }

        public static void Play(bool showOsd = true)
        {
            PlayerControl.PlayMedia(showOsd);
        }

        public static void Pause(bool showOsd = true)
        {
            PlayerControl.PauseMedia(showOsd);
        }

        public static void Stop()
        {
            PlayerControl.StopMedia();
        }

        public static void Close()
        {
            PlayerControl.CloseMedia();
        }

        public static void Seek(long usec)
        {
            PlayerControl.SeekMedia(usec);
        }

        public static void Step()
        {
            PlayerControl.StepMedia();
        }

        public static class Frame
        {
            public static event EventHandler<FrameEventArgs> Decoded
            {
                add { PlayerControl.FrameDecoded += value; }
                remove { PlayerControl.FrameDecoded -= value; }
            }

            public static event EventHandler<FrameEventArgs> Rendered
            {
                add { PlayerControl.FrameRendered += value; }
                remove { PlayerControl.FrameRendered -= value; }
            }

            public static event EventHandler<FrameEventArgs> Presented
            {
                add { PlayerControl.FramePresented += value; }
                remove { PlayerControl.FramePresented -= value; }
            }
        }
    }

    public static class GuiThread
    {
        public static void Do(Action action)
        {
            Gui.VideoBox.Invoke(action);
        }

        public static void DoAsync(Action action)
        {
            Gui.VideoBox.BeginInvoke(action);
        }
    }
}
