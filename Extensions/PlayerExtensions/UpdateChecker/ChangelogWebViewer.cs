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
using System.Runtime.InteropServices;
using System.Windows.Forms;
using Mpdn.Extensions.Framework;

namespace Mpdn.Extensions.PlayerExtensions.UpdateChecker
{
    [ComVisible(true)]
    public class ChangelogWebViewer : WebBrowser
    {
        public class LoadingChangelogEvent : EventArgs
        {
            public List<string> ChangelogLines { get; set; }
        }

        public delegate void PreviousChangelogHandler(object sender, LoadingChangelogEvent args);
        public event PreviousChangelogHandler BeforeLoadPreviousChangelog;

        protected static List<string> HtmlHeaders
        {
            get
            {
                return new List<string>
                {
                    "<!doctype html>",
                    "<html>",
                    "<head>",
                    "<style>" +
                    "body { background: #fff; margin: 0 auto; } " +
                    "h1 { font-size: 15px; color: #1562b6; padding-top: 5px; border: 0px !important; border-bottom: 2px solid #1562b6 !important; }" +
                    "h2 { font-size: 13px; color: #1562b6; padding-top: 5px; border: 0px !important; border-bottom: 1px solid #1562b6 !important; }" +
                    ".center {text-align: center}" +
                    "</style>",
                    "</head>",
                    "<body>"
                };
            }
        }

        public ChangelogWebViewer()
        {
            ObjectForScripting = this;
            IsWebBrowserContextMenuEnabled = false;
            WebBrowserShortcutsEnabled = false;
        }
        /// <summary>
        /// Set the changelog in the WebBrowser
        /// </summary>
        /// <param name="changelogLines"></param>
        public void SetChangelog(IEnumerable<string> changelogLines)
        {
            SetChangelog(changelogLines, false);  
        }

        private void SetChangelog(IEnumerable<string> changelogLines, bool isPrevious)
        {
            var lines = HtmlHeaders;
            lines.AddRange(changelogLines);
            if (!isPrevious)
            {
                lines.Add(
                    "<div class=\"center\"><a href=\"#\" onclick=\"window.external.LoadPreviousChangelog();\">Load previous changelogs</a></div>");
            }
            lines.Add("</body>");
            lines.Add("</html>");
            DocumentText = string.Join("\n", lines);
        }

        public void LoadPreviousChangelog()
        {
            var changelogEvent = new LoadingChangelogEvent();
            BeforeLoadPreviousChangelog.Handle(handler => handler(this, changelogEvent));
            if (changelogEvent.ChangelogLines != null)
            {
                SetChangelog(changelogEvent.ChangelogLines, true);
            }
        }
    }
}