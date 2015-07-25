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
using Mpdn.Extensions.Framework;

namespace Mpdn.Extensions.PlayerExtensions.UpdateChecker
{
    public class WebFile
    {
        public delegate void FileDownloadHandler(object sender);

        public delegate void FileDownloadErrorHandler(object sender, Exception error);

        private readonly WebClient m_WebClient = new WebClient();

        public WebFile(Uri fileUri) : this(fileUri, DefaultFilePath(fileUri))
        {
        }

        public WebFile(Uri fileUri, string filePath)
        {
            if (fileUri == null || filePath == null)
            {
                throw new ArgumentNullException();
            }
            FileUri = fileUri;
            FilePath = filePath;
            m_WebClient.DownloadFileCompleted += WebClientOnDownloadFileCompleted;
            m_WebClient.DownloadProgressChanged +=
                (sender, args) => DownloadProgressChanged.Handle(h => h(sender, args));
        }

        public Uri FileUri { get; private set; }
        public string FilePath { get; private set; }

        private static string DefaultFilePath(Uri fileUri)
        {
            var fi = new FileInfo(fileUri.AbsolutePath);
            return AppPath.GetUserDataFilePath(fi.Name, "Downloads");
        }

        public event FileDownloadHandler Downloaded;
        public event FileDownloadHandler Cancelled;
        public event FileDownloadErrorHandler DownloadFailed;
        public event DownloadProgressChangedEventHandler DownloadProgressChanged;

        private void WebClientOnDownloadFileCompleted(object sender, AsyncCompletedEventArgs asyncCompletedEventArgs)
        {
            if (asyncCompletedEventArgs.Cancelled)
            {
                if (Exists())
                {
                    Delete();
                }
                Cancelled.Handle(h => h(this));
                return;
            }

            if (asyncCompletedEventArgs.Error != null)
            {
                DownloadFailed.Handle(h => h(this, asyncCompletedEventArgs.Error));

                Trace.Write(asyncCompletedEventArgs.Error);
                return;
            }
           

            if (Exists())
            {
                Downloaded.Handle(h => h(this));
            }
        }

        protected void PrepareWebClientRequest()
        {
            WebClientHelper.SetHeaders(m_WebClient);
        }

        public bool Exists()
        {
            return File.Exists(FilePath);
        }

        public void Delete()
        {
            File.Delete(FilePath);
        }

        public Process Start()
        {
            if (!Exists())
            {
                throw new InvalidOperationException("The file to be run doesn't exists");
            }
            return Process.Start(FilePath);
        }

        public void DownloadFile()
        {
            PrepareWebClientRequest();
            Task.Factory.StartNew(() => m_WebClient.DownloadFileAsync(FileUri, FilePath));
        }

        public void CancelDownload()
        {
            m_WebClient.CancelAsync();
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
            {
                throw new ArgumentNullException();
            }

            var path = Path.GetTempPath();
            var fileName = string.Format("{0}{1}", Guid.NewGuid(), extension);
            return Path.Combine(path, fileName);
        }

        private static string GetExtensionFromUrl(Uri fileUri)
        {
            var fi = new FileInfo(fileUri.AbsolutePath);
            var ext = fi.Extension;
            return !string.IsNullOrWhiteSpace(ext) ? ext : null;
        }
    }
}