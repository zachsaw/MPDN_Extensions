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
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Windows.Forms;
using Microsoft.ClearScript;
using Microsoft.ClearScript.Windows;
using SharpDX;
using Color = System.Drawing.Color;
using Point = System.Drawing.Point;
using Rectangle = System.Drawing.Rectangle;

namespace Mpdn.RenderScript
{
    namespace Mpdn.ScriptedRenderChain
    {
        public class ScriptedRenderChain : RenderChain
        {
            #region Settings

            public string ScriptFileName { get; set; }

            #endregion

            private readonly ScriptEngine m_Engine = new JScriptEngine(WindowsScriptEngineFlags.EnableDebugging);
            private readonly HashSet<string> m_FilterTypeNames = new HashSet<string>();
            private readonly HashSet<string> m_EnumTypeNames = new HashSet<string>();
            private string m_RsFile;
            private string m_RsFileName;
            private DateTime m_LastModified = DateTime.MinValue;

            private ITexture m_ErrorText;
            private Font m_ErrorFont;
            private readonly ScriptParser m_ScriptParser;

            protected override string ShaderPath
            {
                get { return "ScriptedRenderChain"; }
            }

            private Font ErrorFont
            {
                get
                {
                    return m_ErrorFont = m_ErrorFont ?? new Font(FontFamily.GenericMonospace, 11, FontStyle.Bold);
                }
            }

            private static string DefaultScriptFileName
            {
                get { return Path.Combine(PathHelper.GetDirectoryName(PlayerControl.ConfigRootPath), "DefaultScript.rs"); }
            }

            public ScriptedRenderChain()
            {
                if (string.IsNullOrWhiteSpace(ScriptFileName))
                {
                    ScriptFileName = DefaultScriptFileName;
                    if (!File.Exists(ScriptFileName))
                    {
                        CreateDefaultScriptFile();
                    }
                }

                m_Engine.AllowReflection = true;
                m_Engine.AddHostObject("__$xhost", new InternalHostFunctions());
                m_Engine.AddHostObject("host", new Host());
                m_Engine.AddHostObject("gpu", Renderer.Dx9GpuInfo.Details);
                m_Engine.AddHostType("Debug", typeof(Debug));

                AddEnumTypes(Assembly.GetAssembly(typeof(IRenderScript)));
                var asm = Assembly.GetExecutingAssembly();
                AddRenderScriptTypes(asm);
                AddEnumTypes(asm);

                m_ScriptParser = new ScriptParser(m_FilterTypeNames);
            }

            public override void RenderScriptDisposed()
            {
                DisposeHelper.Dispose(ref m_ErrorText);
                DisposeHelper.Dispose(ref m_ErrorFont);

                base.RenderScriptDisposed();
            }

            private void CreateDefaultScriptFile()
            {
                File.WriteAllText(ScriptFileName, Helpers.DefaultScript);
            }

            private void AddEnumTypes(Assembly asm)
            {
                var enumTypes = asm.GetTypes().Where(t => t.IsEnum && t.IsPublic);
                foreach (var t in enumTypes)
                {
                    if (m_EnumTypeNames.Contains(t.Name))
                    {
                        throw new Exception(string.Format("Conflicting enum types detected: {0}", t.Name));
                    }
                    m_Engine.AddHostType(t.Name, t);
                    m_EnumTypeNames.Add(t.Name);
                }
            }

            private void AddRenderScriptTypes(Assembly asm)
            {
                var filterTypes =
                    asm.GetTypes()
                        .Where(
                            t =>
                                t.IsSubclassOf(typeof (RenderChain)) && t.IsPublic && !t.IsAbstract &&
                                t.GetConstructor(Type.EmptyTypes) != null);
                foreach (var t in filterTypes)
                {
                    if (m_FilterTypeNames.Contains(t.Name))
                    {
                        throw new Exception(string.Format("Conflicting render script types detected: {0}", t.Name));
                    }
                    m_Engine.AddHostType(t.Name, t);
                    m_FilterTypeNames.Add(t.Name);
                }
            }

            public override IFilter CreateFilter(IFilter input)
            {
                DisposeHelper.Dispose(ref m_ErrorText);
                try
                {
                    m_Engine.CollectGarbage(true);
                    var clip = new Clip(this, input);
                    m_Engine.Script["input"] = clip;
                    m_Engine.Execute("RenderScript", true, BuildScript(ScriptFileName));
                    return clip.Filter;
                }
                catch (Exception ex)
                {
                    return DisplayError(ex);
                }
            }

            private IFilter DisplayError(Exception ex)
            {
                var message = m_Engine.GetStackTrace();
                message = string.Format("Error in render script ('{0}'):\r\n\r\n{1}",
                    m_RsFileName, string.IsNullOrEmpty(message) ? GetInnerMostMessage(ex) : message);
                Trace.WriteLine(message);
                CreateErrorText(message);
                return new ShaderFilter(CompileShader("ErrorTexture.hlsl"), new TextureSourceFilter(m_ErrorText));
            }

            private static string GetInnerMostMessage(Exception e)
            {
                while (e.InnerException != null)
                {
                    e = e.InnerException;
                }
                var scriptEngineException = e as ScriptEngineException;
                if (scriptEngineException != null)
                {
                    return scriptEngineException.ErrorDetails;
                }
                return e.Message;
            }

            private void CreateErrorText(string message)
            {
                var width = Renderer.TargetSize.Width;
                var height = Renderer.TargetSize.Height;
                using (var bmp = new Bitmap(width, height, PixelFormat.Format24bppRgb))
                {
                    var bounds = new Rectangle(0, 0, bmp.Width, bmp.Height);
                    using (var g = Graphics.FromImage(bmp))
                    {
                        g.FillRectangle(Brushes.DarkSlateBlue, bounds);
                        TextRenderer.DrawText(g, message, ErrorFont, new Point(10, 10), Color.OrangeRed);
                    }
                    UpdateErrorTexture(bmp);
                }
            }

            private unsafe void UpdateErrorTexture(Bitmap bmp)
            {
                var width = bmp.Width;
                var height = bmp.Height;
                var bounds = new Rectangle(0, 0, width, height);

                var bmpData = bmp.LockBits(bounds, ImageLockMode.ReadOnly, bmp.PixelFormat);
                try
                {
                    var result = Renderer.CreateTexture(new Size(width, height));
                    m_ErrorText = result;

                    var pitch = width*4;
                    var tex = new Half[height, pitch];
                    var bmpPtr = (byte*) bmpData.Scan0.ToPointer();
                    for (int j = 0; j < height; j++)
                    {
                        byte* ptr = bmpPtr + bmpData.Stride*j;
                        for (int i = 0; i < pitch; i += 4)
                        {
                            tex[j, (i + 3)] = 1; // a
                            tex[j, (i + 2)] = *ptr++/255.0f; // b
                            tex[j, (i + 1)] = *ptr++/255.0f; // g
                            tex[j, (i + 0)] = *ptr++/255.0f; // r
                        }
                    }
                    Renderer.UpdateTexture(result, tex);
                }
                finally
                {
                    bmp.UnlockBits(bmpData);
                }
            }

            private string BuildScript(string scriptRs)
            {
                scriptRs = Path.GetFullPath(scriptRs);

                var lastMod = File.GetLastWriteTimeUtc(scriptRs);
                if (m_RsFileName == scriptRs && lastMod == m_LastModified)
                    return m_RsFile;

                m_RsFile = m_ScriptParser.BuildScript(File.ReadAllText(scriptRs));
                m_RsFileName = scriptRs;
                m_LastModified = lastMod;

                return m_RsFile;
            }
        }

        public class ScriptedRenderChainUi : RenderChainUi<ScriptedRenderChain, ScriptedRenderChainConfigDialog>
        {
            public override string Category
            {
                get { return "Hidden"; } // Change to something else to make it visible in MPDN
            }

            public override ExtensionUiDescriptor Descriptor
            {
                get
                {
                    return new ExtensionUiDescriptor
                    {
                        Name = "Scripted Render Chain",
                        Description = "Write your own render chain using Avisynth-like scripting language",
                        Guid = new Guid("E38CC06E-F1EB-4D57-A01B-C7010D0D9C6A"),
                        Copyright = "" // Optional field
                    };
                }
            }
        }
    }
}
