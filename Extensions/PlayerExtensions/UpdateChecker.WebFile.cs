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
using System.Collections.Specialized;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Threading.Tasks;

namespace Mpdn.Extensions.PlayerExtensions.UpdateChecker
{
    public class WebFile
    {
        public delegate void FileDownloadedHandler(object sender);

        public delegate void FileDownloadErrorHandler(object sender, Exception error);

        private readonly WebClient m_WebClient = new WebClient();

        public WebFile(Uri fileUri) : this(fileUri, DefaultFilePath(fileUri))
        {
        }

        public WebFile(Uri fileUri, string filePath)
        {
            if (fileUri == null || filePath == null)
                throw new ArgumentNullException();
            FileUri = fileUri;
            FilePath = filePath;
            m_WebClient.DownloadFileCompleted += WebClientOnDownloadFileCompleted;
            m_WebClient.DownloadProgressChanged += (sender, args) => DownloadProgressChanged(sender,args);
            HttpHeaders = new NameValueCollection();
        }

        public Uri FileUri { get; private set; }
        public string FilePath { get; private set; }
        public NameValueCollection HttpHeaders { get; set; }

        private static string DefaultFilePath(Uri fileUri)
        {
            var fi = new FileInfo(fileUri.AbsolutePath);
            return AppPath.GetUserDataFilePath(fi.Name, "Downloads");
        }

        public event FileDownloadedHandler Downloaded;
        public event FileDownloadErrorHandler DownloadFailed;
        public event DownloadProgressChangedEventHandler DownloadProgressChanged;

        private void WebClientOnDownloadFileCompleted(object sender, AsyncCompletedEventArgs asyncCompletedEventArgs)
        {
            if (asyncCompletedEventArgs.Error != null)
            {
                if (DownloadFailed != null) 
                    DownloadFailed(this, asyncCompletedEventArgs.Error);

                Trace.Write(asyncCompletedEventArgs.Error);
                return;
            }
            if (Downloaded != null && Exists())
                Downloaded(this);
        }

        protected void PrepareWebClientRequest()
        {
            m_WebClient.Headers.Add(HttpHeaders);
        }

        public bool Exists()
        {
            return File.Exists(FilePath);
        }

        public Process Execute()
        {
            return Process.Start(FilePath);
        }

        public void DownloadFile()
        {
            PrepareWebClientRequest();
            Task.Factory.StartNew(() => m_WebClient.DownloadFileAsync(FileUri, FilePath));
        }
    }

    public class TemporaryWebFile : WebFile
    {
        public TemporaryWebFile(Uri fileUri, string fileExtension)
            : base(fileUri, GetTempFilePathWithExtension(fileExtension))
        {
        }

        public TemporaryWebFile(Uri fileUri)
            : base(fileUri, GetTempFilePathWithExtension(GetExtensionFromUrl(fileUri)))
        {
        }

        private static string GetTempFilePathWithExtension(string extension)
        {
            if (extension == null)
                throw new ArgumentNullException();

            var path = Path.GetTempPath();
            var fileName = string.Format("{0}{1}", Guid.NewGuid(), extension);
            return Path.Combine(path, fileName);
        }

        private static string GetExtensionFromUrl(Uri fileUri)
        {
            var fi = new FileInfo(fileUri.AbsolutePath);
            var ext = fi.Extension;
            if (!string.IsNullOrWhiteSpace(ext))
            {
                return ext;
            }
            return null;
        }
    }
}