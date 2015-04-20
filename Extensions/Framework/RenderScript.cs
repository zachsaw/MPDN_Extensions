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
using System.IO;
using System.Windows.Forms;
using System.Collections.Generic;

namespace Mpdn.RenderScript
{
    public static class RenderScript
    {
        public static IRenderScriptUi Empty = new NullRenderScriptUi();

        public static bool IsEmpty(this IRenderScriptUi script)
        {
            return script is NullRenderScriptUi;
        }

        private class NullRenderScriptUi : IRenderScriptUi
        {
            public ExtensionUiDescriptor Descriptor
            {
                get
                {
                    return new ExtensionUiDescriptor
                    {
                        Guid = Guid.Empty,
                        Name = "None",
                        Description = "Do not use render script"
                    };
                }
            }

            public IRenderScript CreateRenderScript()
            {
                return null;
            }

            public void Initialize()
            {
            }

            public void Destroy()
            {
            }

            public bool HasConfigDialog()
            {
                return false;
            }

            public bool ShowConfigDialog(IWin32Window owner)
            {
                return false;
            }
        }
    }

    public static class ShaderCache<T>
    where T : class
    {
        private static readonly Dictionary<string, ShaderWithDateTime> s_CompiledShaders =
            new Dictionary<string, ShaderWithDateTime>();

        public static T Add(string shaderPath, Func<string, T> compileFunc)
        {
            var lastMod = File.GetLastWriteTimeUtc(shaderPath);

            ShaderWithDateTime result;
            if (s_CompiledShaders.TryGetValue(shaderPath, out result) &&
                result.LastModified == lastMod)
            {
                return result.Shader;
            }

            if (result != null)
            {
                DisposeHelper.Dispose(result.Shader);
                s_CompiledShaders.Remove(shaderPath);
            }

            var shader = compileFunc(shaderPath);
            s_CompiledShaders.Add(shaderPath, new ShaderWithDateTime(shader, lastMod));
            return shader;
        }

        public class ShaderWithDateTime
        {
            public T Shader { get; private set; }
            public DateTime LastModified { get; private set; }

            public ShaderWithDateTime(T shader, DateTime lastModified)
            {
                Shader = shader;
                LastModified = lastModified;
            }
        }
    }

}