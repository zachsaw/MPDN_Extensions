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
using System.Net;
using System.Text;
using Mpdn.Extensions.PlayerExtensions.Exceptions;
using Mpdn.Extensions.Framework;

namespace Mpdn.Extensions.PlayerExtensions
{
    #region Subtitle

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

    #endregion

    public class SubtitleDownloader
    {
        private const string OS_URL = "http://www.opensubtitles.org/isdb/index.php?player=mpc&name[0]={0}&size[0]={1}&hash[0]={2}";

        private const string OS_DL_SUB = "http://www.opensubtitles.org/isdb/dl.php?id={0}&ticket={1}";
        private readonly string m_UserAgent;
        private readonly WebClient m_WebClient = new WebClient();
        private string m_LastTicket;
        private string m_MediaFilename;


        public SubtitleDownloader(string userAgent)
        {
            m_UserAgent = userAgent;
        }

        private string DoRequest(string url)
        {
            try
            {
                using (var client = m_WebClient)
                {
                    client.Headers.Set("User-Agent", m_UserAgent);
                    return client.DownloadString(url);
                }
            }
            catch (Exception)
            {
                throw new InternetConnectivityException();
            }
        }

        /// <summary>
        ///     Get Subtitles for the file
        /// </summary>
        /// <param name="filename"></param>
        /// <returns></returns>
        public List<Subtitle> GetSubtitles(string filename)
        {
            if (string.IsNullOrEmpty(filename))
            {
                throw new ArgumentNullException("filename");
            }
            var file = new FileInfo(filename);
            if (!file.Exists)
            {
                throw new ArgumentException("File doesn't exist", filename);
            }

            var name = file.Name;
            var size = file.Length.ToString("X");
            var hash = HashCalculator.GetHash(filename);
            m_MediaFilename = filename;
            var subs = DoRequest(string.Format(OS_URL, name, size, hash));

            if (string.IsNullOrEmpty(subs))
            {
                throw new EmptyResponseException();
            }

            return ParseSubtitlesResponse(subs);
        }

        private List<Subtitle> ParseSubtitlesResponse(string subs)
        {
            var subList = new List<Subtitle>();
            var subtitle = new Subtitle(this);
            foreach (var line in subs.Split(new[] {"\n"}, StringSplitOptions.RemoveEmptyEntries))
            {
                if (line.StartsWith("ticket="))
                {
                    m_LastTicket = GetValue(line);
                }
                else if (line.StartsWith("movie="))
                {
                    var value = GetValue(line);
                    subtitle.Movie = value.Remove(value.Length - 1);
                }
                else if (line.StartsWith("subtitle="))
                {
                    subtitle.Id = int.Parse(GetValue(line));
                }
                else if (line.StartsWith("language="))
                {
                    subtitle.Lang = GetValue(line);
                }
                else if (line.StartsWith("name="))
                {
                    subtitle.Name = GetValue(line);
                }
                else if (line.Equals("endsubtitle"))
                {
                    subList.Add(subtitle);
                    subtitle = new Subtitle(this);
                }
            }
            return subList;
        }

        private string GetValue(string line)
        {
            return line.Split(new[] {"="}, StringSplitOptions.RemoveEmptyEntries)[1];
        }

        public string FetchSubtitleText(Subtitle subtitle)
        {
            var url = string.Format(OS_DL_SUB, subtitle.Id, m_LastTicket);
            var sub = DoRequest(url);
            if (string.IsNullOrEmpty(sub))
            {
                throw new EmptyResponseException();
            }
            return sub;
        }

        public void SaveSubtitleFile(Subtitle subtitle)
        {
            var dir = PathHelper.GetDirectoryName(m_MediaFilename);
            var subFile = string.Format(Subtitle.FILE_NAME_FORMAT, Path.GetFileNameWithoutExtension(m_MediaFilename),
                subtitle.Lang);
            var fullPath = Path.Combine(dir, subFile);
            var subs = FetchSubtitleText(subtitle);
            if (string.IsNullOrWhiteSpace(subs))
                throw new Exception("Empty Subtitle");
            var subtitleLines = subs.Split(new[] {"\n"}, StringSplitOptions.RemoveEmptyEntries);
            File.WriteAllLines(@fullPath, subtitleLines);
        }
    }

    #region HashCalculator

    public class HashCalculator
    {
        public static string GetHash(string filename)
        {
            return ToHexadecimal(ComputeMovieHash(filename));
        }

        private static byte[] ComputeMovieHash(string filename)
        {
            byte[] result;
            using (Stream input = File.OpenRead(filename))
            {
                result = ComputeMovieHash(input);
            }
            return result;
        }

        private static byte[] ComputeMovieHash(Stream input)
        {
            var streamsize = input.Length;
            var lhash = streamsize;

            long i = 0;
            var buffer = new byte[sizeof (long)];
            while (i < 65536/sizeof (long) && (input.Read(buffer, 0, sizeof (long)) > 0))
            {
                i++;
                lhash += BitConverter.ToInt64(buffer, 0);
            }

            input.Position = Math.Max(0, streamsize - 65536);
            i = 0;
            while (i < 65536/sizeof (long) && (input.Read(buffer, 0, sizeof (long)) > 0))
            {
                i++;
                lhash += BitConverter.ToInt64(buffer, 0);
            }
            input.Close();
            var result = BitConverter.GetBytes(lhash);
            Array.Reverse(result);
            return result;
        }

        private static string ToHexadecimal(byte[] bytes)
        {
            var hexBuilder = new StringBuilder();
            foreach (var data in bytes)
            {
                hexBuilder.Append(data.ToString("x2"));
            }
            return hexBuilder.ToString();
        }
    }

    #endregion
}