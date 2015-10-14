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
using System.Collections.Generic;
using System.Windows.Forms;
using MediaInfoDotNet;
using MediaInfoLib;
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

            var mediaFile = new MediaFile(path);
            var mi = new MediaInfo();
            mi.Open(path);

            var lines = new List<string>();
            {
                lines.Add("<!doctype html>");
                lines.Add("<html>");
                lines.Add("<head>");
                lines.Add("<style>* { padding: 0px; margin: 0px; font-family:tahoma; } body { width: 700px; background: #fff; margin: 0 auto; } table { width: 700px; font-size: 12px; border-collapse: collapse; } td { width: 570px; padding: 5px; border-bottom: 1px dotted #1562b6; word-wrap: break-word; } td:first-child { width: 130px; border-right: 1px solid #1562b6; } .thead { font-size: 15px; color: #1562b6; padding-top: 25px; border: 0px !important; border-bottom: 2px solid #1562b6 !important; } .no-padding { padding-top: 0px !important; }</style>");
                lines.Add("</head>");
                lines.Add("<body>");
                lines.Add("<table border='0' cellspacing='0' cellpadding='0'>");
                lines.Add("<tr><td class='thead no-padding' colspan='2'><b>General</b></td></tr>");
                lines.Add(string.IsNullOrEmpty(mediaFile.miGetString("UniqueID/String")) ? string.Empty : string.Format("<tr><td>Unique ID:</td><td>{0}</td></tr>", mediaFile.miGetString("UniqueID/String")));
                lines.Add(string.IsNullOrEmpty(mediaFile.miGetString("Movie")) ? string.Empty : string.Format("<tr><td>Movie name:</td><td>{0}</td></tr>", mediaFile.miGetString("Movie")));
                lines.Add(string.IsNullOrEmpty(mediaFile.miGetString("CompleteName")) ? string.Empty : string.Format("<tr><td>Complete name:</td><td>{0}</td></tr>", mediaFile.miGetString("CompleteName")));
                lines.Add(string.IsNullOrEmpty(mediaFile.miGetString("Format")) ? string.Empty : string.Format("<tr><td>Format:</td><td>{0}</td></tr>", mediaFile.miGetString("Format")));
                lines.Add(string.IsNullOrEmpty(mediaFile.miGetString("Format_Version")) ? string.Empty : string.Format("<tr><td>Format version:</td><td>{0}</td></tr>", mediaFile.miGetString("Format_Version")));
                lines.Add(string.IsNullOrEmpty(mediaFile.miGetString("FileSize/String")) ? string.Empty : string.Format("<tr><td>File size:</td><td>{0}</td></tr>", mediaFile.miGetString("FileSize/String")));
                lines.Add(string.IsNullOrEmpty(mediaFile.miGetString("Duration/String")) ? string.Empty : string.Format("<tr><td>Duration:</td><td>{0}</td></tr>", mediaFile.miGetString("Duration/String")));
                lines.Add(string.IsNullOrEmpty(mediaFile.miGetString("OverallBitRate/String")) ? string.Empty : string.Format("<tr><td>Overall bitrate:</td><td>{0}</td></tr>", mediaFile.miGetString("OverallBitRate/String")));
                lines.Add(string.IsNullOrEmpty(mediaFile.miGetString("Encoded_Date")) ? string.Empty : string.Format("<tr><td>Encoded date:</td><td>{0}</td></tr>", mediaFile.miGetString("Encoded_Date")));
                lines.Add(string.IsNullOrEmpty(mediaFile.miGetString("Encoded_Application")) ? string.Empty : string.Format("<tr><td>Writing application:</td><td>{0}</td></tr>", mediaFile.miGetString("Encoded_Application")));
                lines.Add(string.IsNullOrEmpty(mediaFile.miGetString("Encoded_Library/String")) ? string.Empty : string.Format("<tr><td>Writing library:</td><td>{0}</td></tr>", mediaFile.miGetString("Encoded_Library/String")));
            }

            lines.Add("<tr><td class='thead' colspan='2'><b>Video</b></td></tr>");
            foreach (var info in mediaFile.Video)
            {
                lines.Add(string.Format("<tr><td>ID:</td><td>{0}</td></tr>", info.Value.miGetString("ID/String")));
                lines.Add(string.IsNullOrEmpty(info.Value.miGetString("Title")) ? string.Empty : string.Format("<tr><td>Title:</td><td>{0}</td></tr>", info.Value.miGetString("Title")));
                lines.Add(string.IsNullOrEmpty(info.Value.miGetString("Format")) ? string.Empty : string.Format("<tr><td>Format:</td><td>{0}</td></tr>", info.Value.miGetString("Format")));
                lines.Add(string.IsNullOrEmpty(info.Value.miGetString("Format/Info")) ? string.Empty : string.Format("<tr><td>Format info:</td><td>{0}</td></tr>", info.Value.miGetString("Format/Info")));
                lines.Add(string.IsNullOrEmpty(info.Value.miGetString("Format_Profile")) ? string.Empty : string.Format("<tr><td>Format profile:</td><td>{0}</td></tr>", info.Value.miGetString("Format_Profile")));
                lines.Add(string.IsNullOrEmpty(info.Value.miGetString("Format_Settings")) ? string.Empty : string.Format("<tr><td>Format settings:</td><td>{0}</td></tr>", info.Value.miGetString("Format_Settings")));
                lines.Add(string.IsNullOrEmpty(info.Value.miGetString("MuxingMode")) ? string.Empty : string.Format("<tr><td>Muxing mode:</td><td>{0}</td></tr>", info.Value.miGetString("MuxingMode")));
                lines.Add(string.IsNullOrEmpty(info.Value.miGetString("CodecID/String")) ? string.Empty : string.Format("<tr><td>Codec ID:</td><td>{0}</td></tr>", info.Value.miGetString("CodecID/String")));
                lines.Add(string.IsNullOrEmpty(info.Value.miGetString("Duration/String")) ? string.Empty : string.Format("<tr><td>Duration:</td><td>{0}</td></tr>", info.Value.miGetString("Duration/String")));
                lines.Add(string.IsNullOrEmpty(info.Value.miGetString("BitRate/String")) ? string.Empty : string.Format("<tr><td>Bitrate:</td><td>{0}</td></tr>", info.Value.miGetString("BitRate/String")));
                lines.Add(string.IsNullOrEmpty(info.Value.miGetString("BitRate_Nominal/String")) ? string.Empty : string.Format("<tr><td>Nominal bitrate:</td><td>{0}</td></tr>", info.Value.miGetString("BitRate_Nominal/String")));
                lines.Add(string.IsNullOrEmpty(info.Value.miGetString("Width/String")) ? string.Empty : string.Format("<tr><td>Width:</td><td>{0}</td></tr>", info.Value.miGetString("Width/String")));
                lines.Add(string.IsNullOrEmpty(info.Value.miGetString("Height/String")) ? string.Empty : string.Format("<tr><td>Height:</td><td>{0}</td></tr>", info.Value.miGetString("Height/String")));
                lines.Add(string.IsNullOrEmpty(info.Value.miGetString("DisplayAspectRatio/String")) ? string.Empty : string.Format("<tr><td>Display aspect ratio:</td><td>{0}</td></tr>", info.Value.miGetString("DisplayAspectRatio/String")));
                lines.Add(string.IsNullOrEmpty(info.Value.miGetString("FrameRate_Mode/String")) ? string.Empty : string.Format("<tr><td>Frame rate mode:</td><td>{0}</td></tr>", info.Value.miGetString("FrameRate_Mode/String")));
                lines.Add(string.IsNullOrEmpty(info.Value.miGetString("FrameRate/String")) ? string.Empty : string.Format("<tr><td>Frame rate:</td><td>{0}</td></tr>", info.Value.miGetString("FrameRate/String")));
                lines.Add(string.IsNullOrEmpty(info.Value.miGetString("ColorSpace")) ? string.Empty : string.Format("<tr><td>Color space:</td><td>{0}</td></tr>", info.Value.miGetString("ColorSpace")));
                lines.Add(string.IsNullOrEmpty(info.Value.miGetString("ChromaSubsampling")) ? string.Empty : string.Format("<tr><td>Chroma subsampling:</td><td>{0}</td></tr>", info.Value.miGetString("ChromaSubsampling")));
                lines.Add(string.IsNullOrEmpty(info.Value.miGetString("BitDepth/String")) ? string.Empty : string.Format("<tr><td>Bit depth:</td><td>{0}</td></tr>", info.Value.miGetString("BitDepth/String")));
                lines.Add(string.IsNullOrEmpty(info.Value.miGetString("ScanType")) ? string.Empty : string.Format("<tr><td>Scan type:</td><td>{0}</td></tr>", info.Value.miGetString("ScanType")));
                lines.Add(string.IsNullOrEmpty(info.Value.miGetString("Bits-(Pixel*Frame)")) ? string.Empty : string.Format("<tr><td>Bits/(Pixel*Frame):</td><td>{0}</td></tr>", info.Value.miGetString("Bits-(Pixel*Frame)")));
                lines.Add(string.IsNullOrEmpty(info.Value.miGetString("StreamSize/String")) ? string.Empty : string.Format("<tr><td>Stream size:</td><td>{0}</td></tr>", info.Value.miGetString("StreamSize/String")));
                lines.Add(string.IsNullOrEmpty(info.Value.miGetString("Encoded_Library/String")) ? string.Empty : string.Format("<tr><td>Writing library:</td><td>{0}</td></tr>", info.Value.miGetString("Encoded_Library/String")));
                lines.Add(string.IsNullOrEmpty(info.Value.miGetString("Encoded_Library_Settings")) ? string.Empty : string.Format("<tr><td>Encoding settings:</td><td>{0}</td></tr>", info.Value.miGetString("Encoded_Library_Settings")));
                lines.Add(string.IsNullOrEmpty(info.Value.miGetString("Language/String")) ? string.Empty : string.Format("<tr><td>Language:</td><td>{0}</td></tr>", info.Value.miGetString("Language/String")));
                lines.Add(string.IsNullOrEmpty(info.Value.miGetString("Default/String")) ? string.Empty : string.Format("<tr><td>Default:</td><td>{0}</td></tr>", info.Value.miGetString("Default/String")));
                lines.Add(string.IsNullOrEmpty(info.Value.miGetString("Forced/String")) ? string.Empty : string.Format("<tr><td>Forced:</td><td>{0}</td></tr>", info.Value.miGetString("Forced/String")));
                lines.Add(string.IsNullOrEmpty(info.Value.miGetString("colour_range")) ? string.Empty : string.Format("<tr><td>Color range:</td><td>{0}</td></tr>", info.Value.miGetString("colour_range")));
                lines.Add(string.IsNullOrEmpty(info.Value.miGetString("colour_primaries")) ? string.Empty : string.Format("<tr><td>Color primaries:</td><td>{0}</td></tr>", info.Value.miGetString("colour_primaries")));
                lines.Add(string.IsNullOrEmpty(info.Value.miGetString("transfer_characteristics")) ? string.Empty : string.Format("<tr><td>Transfer characteristics:</td><td>{0}</td></tr>", info.Value.miGetString("transfer_characteristics")));
                lines.Add(string.IsNullOrEmpty(info.Value.miGetString("matrix_coefficients")) ? string.Empty : string.Format("<tr><td>Matrix coefficients:</td><td>{0}</td></tr>", info.Value.miGetString("matrix_coefficients")));
            }

            if (mediaFile.Audio.Count > 0)
            {
                int audioTracks = 1;

                foreach (var info in mediaFile.Audio)
                {
                    lines.Add(mediaFile.Audio.Count == 1 ? "<tr><td class='thead' colspan='2'><b>Audio</b></td></tr>" : string.Format("<tr><td class='thead' colspan='2'><b>Audio #{0}</b></td></tr>", audioTracks));
                    lines.Add(string.Format("<tr><td>ID:</td><td>{0}</td></tr>", info.Value.miGetString("ID/String")));
                    lines.Add(string.IsNullOrEmpty(info.Value.miGetString("Title")) ? string.Empty : string.Format("<tr><td>Title:</td><td>{0}</td></tr>", info.Value.miGetString("Title")));
                    lines.Add(string.IsNullOrEmpty(info.Value.miGetString("Format")) ? string.Empty : string.Format("<tr><td>Format:</td><td>{0}</td></tr>", info.Value.miGetString("Format")));
                    lines.Add(string.IsNullOrEmpty(info.Value.miGetString("Format/Info")) ? string.Empty : string.Format("<tr><td>Format info:</td><td>{0}</td></tr>", info.Value.miGetString("Format/Info")));
                    lines.Add(string.IsNullOrEmpty(info.Value.miGetString("CodecID/String")) ? string.Empty : string.Format("<tr><td>Codec ID:</td><td>{0}</td></tr>", info.Value.miGetString("CodecID/String")));
                    lines.Add(string.IsNullOrEmpty(info.Value.miGetString("Duration/String")) ? string.Empty : string.Format("<tr><td>Duration:</td><td>{0}</td></tr>", info.Value.miGetString("Duration/String")));
                    lines.Add(string.IsNullOrEmpty(info.Value.miGetString("BitRate_Mode/String")) ? string.Empty : string.Format("<tr><td>Bitrate mode:</td><td>{0}</td></tr>", info.Value.miGetString("BitRate_Mode/String")));
                    lines.Add(string.IsNullOrEmpty(info.Value.miGetString("BitRate/String")) ? string.Empty : string.Format("<tr><td>Bitrate:</td><td>{0}</td></tr>", info.Value.miGetString("BitRate/String")));
                    lines.Add(string.IsNullOrEmpty(info.Value.miGetString("Channel(s)/String")) ? string.Empty : string.Format("<tr><td>Channels:</td><td>{0}</td></tr>", info.Value.miGetString("Channel(s)/String")));
                    lines.Add(string.IsNullOrEmpty(info.Value.miGetString("SamplingRate/String")) ? string.Empty : string.Format("<tr><td>Sampling rate:</td><td>{0}</td></tr>", info.Value.miGetString("SamplingRate/String")));
                    lines.Add(string.IsNullOrEmpty(info.Value.miGetString("BitDepth/String")) ? string.Empty : string.Format("<tr><td>Bit depth:</td><td>{0}</td></tr>", info.Value.miGetString("BitDepth/String")));
                    lines.Add(string.IsNullOrEmpty(info.Value.miGetString("Compression_Mode/String")) ? string.Empty : string.Format("<tr><td>Compression mode:</td><td>{0}</td></tr>", info.Value.miGetString("Compression_Mode/String")));
                    lines.Add(string.IsNullOrEmpty(info.Value.miGetString("StreamSize/String")) ? string.Empty : string.Format("<tr><td>Stream size:</td><td>{0}</td></tr>", info.Value.miGetString("StreamSize/String")));
                    lines.Add(string.IsNullOrEmpty(info.Value.miGetString("Language/String")) ? string.Empty : string.Format("<tr><td>Language:</td><td>{0}</td></tr>", info.Value.miGetString("Language/String")));
                    lines.Add(string.IsNullOrEmpty(info.Value.miGetString("Default/String")) ? string.Empty : string.Format("<tr><td>Default:</td><td>{0}</td></tr>", info.Value.miGetString("Default/String")));
                    lines.Add(string.IsNullOrEmpty(info.Value.miGetString("Forced/String")) ? string.Empty : string.Format("<tr><td>Forced:</td><td>{0}</td></tr>", info.Value.miGetString("Forced/String")));

                    audioTracks++;
                }
            }

            if (mediaFile.Text.Count > 0)
            {
                int textTracks = 1;

                foreach (var info in mediaFile.Text)
                {
                    lines.Add(mediaFile.Text.Count == 1 ? "<tr><td class='thead' colspan='2'><b>Text</b></td></tr>" : string.Format("<tr><td class='thead' colspan='2'><b>Text #{0}</b></td></tr>", textTracks));
                    lines.Add(string.Format("<tr><td>ID:</td><td>{0}</td></tr>", info.Value.miGetString("ID/String")));
                    lines.Add(string.IsNullOrEmpty(info.Value.miGetString("Format")) ? string.Empty : string.Format("<tr><td>Format:</td><td>{0}</td></tr>", info.Value.miGetString("Format")));
                    lines.Add(string.IsNullOrEmpty(info.Value.miGetString("CodecID")) ? string.Empty : string.Format("<tr><td>Codec ID:</td><td>{0}</td></tr>", info.Value.miGetString("CodecID")));
                    lines.Add(string.IsNullOrEmpty(info.Value.miGetString("CodecID/Info")) ? string.Empty : string.Format("<tr><td>Codec ID info:</td><td>{0}</td></tr>", info.Value.miGetString("CodecID/Info")));
                    lines.Add(string.IsNullOrEmpty(info.Value.miGetString("Compression_Mode/String")) ? string.Empty : string.Format("<tr><td>Compression mode:</td><td>{0}</td></tr>", info.Value.miGetString("Compression_Mode/String")));
                    lines.Add(string.IsNullOrEmpty(info.Value.miGetString("Language/String")) ? string.Empty : string.Format("<tr><td>Language:</td><td>{0}</td></tr>", info.Value.miGetString("Language/String")));
                    lines.Add(string.IsNullOrEmpty(info.Value.miGetString("Default/String")) ? string.Empty : string.Format("<tr><td>Default:</td><td>{0}</td></tr>", info.Value.miGetString("Default/String")));
                    lines.Add(string.IsNullOrEmpty(info.Value.miGetString("Forced/String")) ? string.Empty : string.Format("<tr><td>Forced:</td><td>{0}</td></tr>", info.Value.miGetString("Forced/String")));

                    textTracks++;
                }
            }

            if (mediaFile.Menu.Count > 0)
            {
                int chapterBegin = int.Parse(mi.Get(StreamKind.Menu, 0, "Chapters_Pos_Begin"));
                int chapterEnd = int.Parse(mi.Get(StreamKind.Menu, 0, "Chapters_Pos_End"));

                lines.Add("<tr><td class='thead' colspan='2'><b>Menu</b></td></tr>");
                foreach (var info in mediaFile.Menu)
                {
                    for (int i = chapterBegin; i < chapterEnd; i++)
                    {
                        string key = mi.Get(StreamKind.Menu, 0, i, InfoKind.Name);
                        string value = mi.Get(StreamKind.Menu, 0, i);
                        lines.Add(string.IsNullOrEmpty(info.Value.miGetString(key)) ? string.Empty : string.Format("<tr><td>{0}:</td><td>{1}</td></tr>", key, value));
                    }
                }
            }

            lines.Add("</table></body></html>");

            wb_info.DocumentText = string.Join("\n", lines.ToArray());
        }

        private void OnBrowserPreviewKeyDown(object sender, PreviewKeyDownEventArgs e)
        {
            if (e.KeyData == Keys.Escape) Close();
        }

        private void ViewMediaInfoFormShown(object sender, System.EventArgs e)
        {
            // Workaround media info form going behind player when it's set to TopMost
            // For some reason, using the web browser can cause this buggy behavior

            var form = Gui.VideoBox.FindForm();
            if (form == null)
                return;

            TopMost = form.TopMost;
        }
    }
}
