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
using System.Windows.Forms;
using MediaInfoDotNet;
using Mpdn.Extensions.Framework;

namespace Mpdn.Extensions.PlayerExtensions
{
    public partial class ViewMediaInfoForm : Form
    {
        public ViewMediaInfoForm(string path)
        {
            InitializeComponent();

            Icon = Gui.Icon;

            wb_info.PreviewKeyDown += OnBrowserPreviewKeyDown;

            int trackId = 1;
            var mediaFile = new MediaFile(path);
            var mediaDuration = TimeSpan.FromMilliseconds(mediaFile.duration);

            var lines = new List<string>();
            {
                lines.Add("<!doctype html>");
                lines.Add("<html>");
                lines.Add("<head>");
                lines.Add("<style>* { padding: 0px; margin: 0px; font-family:tahoma; } body { width: 700px; background: #fff; margin: 0 auto; padding-bottom: 30px; } table { width: 700px; font-size: 12px; border-collapse: collapse; } td { padding: 5px; border-bottom: 1px dotted #1c7feb; } td:first-child { width: 100px; border-right: 1px solid #1c7feb; } .thead { font-size: 15px; color: #1c7feb; padding-top: 25px; border: 0px !important; border-bottom: 2px solid #1c7feb !important; }</style>");
                lines.Add("</head>");
                lines.Add("<body>");
                lines.Add("<table border='0' cellspacing='0' cellpadding='0'>");
                lines.Add("<tr><td class='thead' colspan='2'><b>General</b></td></tr>");
                lines.Add(string.IsNullOrEmpty(mediaFile.uniqueId) ? string.Empty : string.Format("<tr><td>Unique ID:</td><td>{0}</td></tr>", mediaFile.uniqueId));
                lines.Add(string.Format("<tr><td>ID:</td><td>{0}</td></tr>", trackId));
                lines.Add(string.Format("<tr><td>Complete name:</td><td>{0}</td></tr>", path));
                lines.Add(string.Format("<tr><td>Format:</td><td>{0}</td></tr>", mediaFile.format));
                lines.Add(mediaFile.size == 0 ? string.Empty : string.Format("<tr><td>File size:</td><td>{0}</td></tr>", BytesToString(mediaFile.size)));
                lines.Add(string.Format("<tr><td>Duration:</td><td>{0}</td></tr>", mediaDuration.ToString(@"hh\:mm\:ss")));
                lines.Add(string.Format("<tr><td>Overall bit rate:</td><td>{0} Kbps</td></tr>", mediaFile.bitRate / 1000));
                lines.Add(string.Format("<tr><td>Encoded date:</td><td>{0}</td></tr>", mediaFile.encodedDate));
                lines.Add(string.Format("<tr><td>Writing application:</td><td>{0}</td></tr>", mediaFile.miGetString("Encoded_Application")));
                lines.Add(string.IsNullOrEmpty(mediaFile.encodedLibrary) ? string.Empty : string.Format("<tr><td>Writing library:</td><td>{0}</td></tr>", mediaFile.encodedLibrary));
            }

            lines.Add("<tr><td class='thead' colspan='2'><b>Video</b></td></tr>");
            foreach (var info in mediaFile.Video)
            {
                trackId++;

                long videoStreamSize;
                long.TryParse(info.Value.miGetString("StreamSize"), out videoStreamSize);

                var timespan = TimeSpan.FromMilliseconds(info.Value.duration);

                int gcd = GreatestCommonDivisor(info.Value.width, info.Value.height);
                string aspectRatio = info.Value.width / gcd + ":" + info.Value.height / gcd;

                double bitPerFrame = info.Value.size * 8 /
                                     ((double)info.Value.width * info.Value.height * info.Value.frameCount);
                lines.Add(string.Format("<tr><td>ID:</td><td>{0}</td></tr>", trackId));
                lines.Add(string.IsNullOrEmpty(info.Value.format) ? string.Empty : string.Format("<tr><td>Format:</td><td>{0}</td></tr>", info.Value.format));
                lines.Add(string.IsNullOrEmpty(info.Value.miGetString("Format_Profile")) ? string.Empty : string.Format("<tr><td>Format profile:</td><td>{0}</td></tr>", info.Value.miGetString("Format_Profile")));
                lines.Add(string.IsNullOrEmpty(info.Value.miGetString("Format_Settings")) ? string.Empty : string.Format("<tr><td>Format settings:</td><td>{0}</td></tr>", info.Value.miGetString("Format_Settings")));
                lines.Add(string.IsNullOrEmpty(info.Value.muxingMode) ? string.Empty : string.Format("<tr><td>Muxing mode:</td><td>{0}</td></tr>", info.Value.muxingMode));
                lines.Add(string.IsNullOrEmpty(info.Value.codecId) ? string.Empty : string.Format("<tr><td>Codec ID:</td><td>{0}</td></tr>", info.Value.codecId));
                lines.Add(string.Format("<tr><td>Duration:</td><td>{0}</td></tr>", timespan.ToString(@"hh\:mm\:ss")));
                lines.Add(info.Value.bitRate == 0 ? string.Empty : string.Format("<tr><td>Bit rate:</td><td>{0} Kbps</td></tr>", info.Value.bitRate / 1000));
                lines.Add(info.Value.bitRateNominal == 0 ? string.Empty : string.Format("<tr><td>Nominal bit rate:</td><td>{0} Kbps</td></tr>", info.Value.bitRateNominal / 1000));
                lines.Add(info.Value.width == 0 ? string.Empty : string.Format("<tr><td>Width:</td><td>{0} pixels</td></tr>", info.Value.width));
                lines.Add(info.Value.height == 0 ? string.Empty : string.Format("<tr><td>Height:</td><td>{0} pixels</td></tr>", info.Value.height));
                lines.Add(string.IsNullOrEmpty(aspectRatio) ? string.Empty : string.Format("<tr><td>Aspect ratio:</td><td>{0}</td></tr>", aspectRatio));
                lines.Add(string.IsNullOrEmpty(info.Value.frameRateMode) ? string.Empty : string.Format("<tr><td>Frame rate mode:</td><td>{0}</td></tr>", info.Value.frameRateMode));
                lines.Add(info.Value.frameRate == 0 ? string.Empty : string.Format("<tr><td>Frame rate:</td><td>{0} fps</td></tr>", info.Value.frameRate));
                lines.Add(string.IsNullOrEmpty(info.Value.miGetString("ChromaSubsampling")) ? string.Empty : string.Format("<tr><td>Chroma subsampling:</td><td>{0}</td></tr>", info.Value.miGetString("ChromaSubsampling")));
                lines.Add(info.Value.bitDepth == 0 ? string.Empty : string.Format("<tr><td>Bit depth:</td><td>{0} bits</td></tr>", info.Value.bitDepth));
                lines.Add(string.IsNullOrEmpty(info.Value.miGetString("ScanType")) ? string.Empty : string.Format("<tr><td>Scan type:</td><td>{0}</td></tr>", info.Value.miGetString("ScanType")));
                lines.Add(bitPerFrame == 0 ? string.Empty : string.Format("<tr><td>Bits/(Pixel*Frame):</td><td>{0}</td></tr>", bitPerFrame.ToString("0.###")));
                lines.Add(videoStreamSize == 0 ? string.Empty : string.Format("<tr><td>Stream size:</td><td>{0}</td></tr>", BytesToString(videoStreamSize)));
                lines.Add(string.IsNullOrEmpty(info.Value.encodedLibrary) ? string.Empty : string.Format("<tr><td>Writing library:</td><td>{0}</td></tr>", info.Value.encodedLibrary));
                lines.Add(string.IsNullOrEmpty(info.Value.encoderSettingsRaw) ? string.Empty : string.Format("<tr><td>Encoding settings:</td><td>{0}</td></tr>", info.Value.encoderSettingsRaw));
                lines.Add(string.IsNullOrEmpty(info.Value.language) ? string.Empty : string.Format("<tr><td>Language:</td><td>{0}</td></tr>", info.Value.language));
            }

            if (mediaFile.Audio.Count > 0)
            {
                int audioTracks = 1;

                foreach (var info in mediaFile.Audio)
                {
                    trackId++;

                    long audioStreamSize;
                    long.TryParse(info.Value.miGetString("StreamSize"), out audioStreamSize);

                    var timespan = TimeSpan.FromMilliseconds(info.Value.duration);

                    lines.Add(mediaFile.Audio.Count == 1
                        ? "<tr><td class='thead' colspan='2'><b>Audio</b></td></tr>"
                        : string.Format("<tr><td class='thead' colspan='2'><b>Audio #{0}</b></td></tr>", audioTracks));
                    lines.Add(string.Format("<tr><td>ID:</td><td>{0}</td></tr>", trackId));
                    lines.Add(string.IsNullOrEmpty(info.Value.format) ? string.Empty : string.Format("<tr><td>Format:</td><td>{0}</td></tr>", info.Value.format));
                    lines.Add(string.IsNullOrEmpty(info.Value.codecId) ? string.Empty : string.Format("<tr><td>Codec ID:</td><td>{0}</td></tr>", info.Value.codecId));
                    lines.Add(timespan.TotalMilliseconds == 0 ? string.Empty : string.Format("<tr><td>Duration:</td><td>{0}</td></tr>", timespan.ToString(@"hh\:mm\:ss")));
                    lines.Add(string.IsNullOrEmpty(info.Value.bitRateMode) ? string.Empty : string.Format("<tr><td>Bit rate mode:</td><td>{0}</td></tr>", info.Value.bitRateMode));
                    lines.Add(info.Value.bitRate == 0 ? string.Empty : string.Format("<tr><td>Bit rate:</td><td>{0} Kbps</td></tr>", info.Value.bitRate / 1000));
                    lines.Add(string.Format("<tr><td>Channel(s):</td><td>{0} channels</td></tr>", info.Value.channels));
                    lines.Add(info.Value.sampleRate == 0 ? string.Empty : string.Format("<tr><td>Sampling rate:</td><td>{0} KHz</td></tr>", info.Value.sampleRate));
                    lines.Add(info.Value.bitDepth == 0 ? string.Empty : string.Format("<tr><td>Bit depth:</td><td>{0} bits</td></tr>", info.Value.bitDepth));
                    lines.Add(string.IsNullOrEmpty(info.Value.compressionMode) ? string.Empty : string.Format("<tr><td>Compression mode:</td><td>{0}</td></tr>", info.Value.compressionMode));
                    lines.Add(audioStreamSize == 0 ? string.Empty : string.Format("<tr><td>Stream size:</td><td>{0}</td></tr>", BytesToString(audioStreamSize)));
                    lines.Add(string.IsNullOrEmpty(info.Value.language) ? string.Empty : string.Format("<tr><td>Language:</td><td>{0}</td></tr>", info.Value.language));

                    audioTracks++;
                }
            }

            if (mediaFile.Text.Count > 0)
            {
                lines.Add("<tr><td class='thead' colspan='2'><b>Subtitle</b></td></tr>");
                foreach (var info in mediaFile.Text)
                {
                    trackId++;

                    lines.Add(string.Format("<tr><td>ID:</td><td>{0}</td></tr>", trackId));
                    lines.Add(string.IsNullOrEmpty(info.Value.format) ? string.Empty : string.Format("<tr><td>Format:</td><td>{0}</td></tr>", info.Value.format));
                    lines.Add(string.IsNullOrEmpty(info.Value.codecId) ? string.Empty : string.Format("<tr><td>Codec ID:</td><td>{0}</td></tr>", info.Value.codecId));
                    lines.Add(string.IsNullOrEmpty(info.Value.compressionMode) ? string.Empty : string.Format("<tr><td>Compression mode:</td><td>{0}</td></tr>", info.Value.compressionMode));
                    lines.Add(string.IsNullOrEmpty(info.Value.language) ? string.Empty : string.Format("<tr><td>Language:</td><td>{0}</td></tr>", info.Value.language));
                }
            }

            lines.Add("</table></body></html>");

            wb_info.DocumentText = string.Join("\n", lines.ToArray());
        }

        private void OnBrowserPreviewKeyDown(object sender, PreviewKeyDownEventArgs e)
        {
            if (e.KeyData == Keys.Escape) Close();
        }

        private int GreatestCommonDivisor(int a, int b)
        {
            return (b == 0) ? a : GreatestCommonDivisor(b, a % b);
        }

        private string BytesToString(long byteCount)
        {
            string[] suf = { "B", "KB", "MB", "GB", "TB", "PB", "EB" }; //Longs run out around EB
            if (byteCount == 0)
                return "0" + suf[0];
            long bytes = Math.Abs(byteCount);
            int place = Convert.ToInt32(Math.Floor(Math.Log(bytes, 1024)));
            double num = Math.Round(bytes / Math.Pow(1024, place), 2);
            return (Math.Sign(byteCount) * num).ToString() + " " + suf[place];
        }
    }
}
