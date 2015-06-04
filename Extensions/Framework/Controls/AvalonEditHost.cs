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
//css_reference WindowsFormsIntegration;
//css_reference PresentationFramework;
//css_reference PresentationCore;
//css_reference WindowsBase;
//css_reference System.Xaml;

using System.ComponentModel;
using System.ComponentModel.Design.Serialization;
using System.IO;
using System.Reflection;
using System.Windows;
using System.Xml;
using ICSharpCode.AvalonEdit;
using ICSharpCode.AvalonEdit.Highlighting;

namespace Mpdn.Extensions.Framework.Controls
{
    [Designer("System.Windows.Forms.Design.ControlDesigner, System.Design")]
    [DesignerSerializer("System.ComponentModel.Design.Serialization.TypeCodeDomSerializer , System.Design",
        "System.ComponentModel.Design.Serialization.CodeDomSerializer, System.Design")]
    public class AvalonEditHost : System.Windows.Forms.Integration.ElementHost
    {
        private readonly TextEditor m_AvalonEdit = new TextEditor();

        public TextEditor Editor
        {
            get { return m_AvalonEdit; }
        }

        public AvalonEditHost()
        {
            Child = m_AvalonEdit;
            m_AvalonEdit.HorizontalAlignment = HorizontalAlignment.Stretch;
            m_AvalonEdit.VerticalAlignment = VerticalAlignment.Stretch;

            m_AvalonEdit.ShowLineNumbers = true;
            m_AvalonEdit.FontFamily = new System.Windows.Media.FontFamily("Consolas");
            m_AvalonEdit.FontSize = 12.75f;

            LoadSyntaxHighlighter();
        }

        private void LoadSyntaxHighlighter()
        {
            if (LicenseManager.UsageMode == LicenseUsageMode.Designtime)
                return;

            var asm = Assembly.GetAssembly(typeof (TextEditor));
            var path = PathHelper.GetDirectoryName(asm.Location);
            var xshd = Path.Combine(path, "JavaScript-Mode.xshd");

            using (var stream = File.OpenRead(xshd))
            {
                using (var xr = new XmlTextReader(stream))
                {
                    m_AvalonEdit.SyntaxHighlighting =
                        ICSharpCode.AvalonEdit.Highlighting.Xshd.HighlightingLoader.Load(xr,
                            HighlightingManager.Instance);
                }
            }
        }
    }
}
