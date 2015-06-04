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
using System.Collections.Generic;
using System.Runtime.Serialization.Formatters.Binary;
using Mpdn.OpenCl;
using Mpdn.RenderScript;

namespace Mpdn.Extensions.Framework
{
    public static class ShaderCache
    {
        private static bool s_Loaded;

        private static class Cache<T> where T : class, IShaderBase
        {
            private static readonly Dictionary<string, ShaderWithDateTime> s_LoadedShaders =
                new Dictionary<string, ShaderWithDateTime>();

            private static Dictionary<string, ShaderWithDateTime> s_CompiledShaders =
                new Dictionary<string, ShaderWithDateTime>();

            public static bool Modified { get; private set; }

            public static string ShaderCacheRoot
            {
                get { return AppPath.GetUserDataDir("ShaderCache"); }
            }

            public static T AddLoaded(string shaderPath, Func<string, T> loadFunc)
            {
                var lastMod = File.GetLastWriteTimeUtc(shaderPath);

                ShaderWithDateTime result;
                if (s_LoadedShaders.TryGetValue(shaderPath, out result) &&
                    result.LastModified == lastMod)
                {
                    return result.Shader;
                }

                if (result != null)
                {
                    DisposeHelper.Dispose(result.Shader);
                    s_LoadedShaders.Remove(shaderPath);
                }

                var shader = loadFunc(shaderPath);
                s_LoadedShaders.Add(shaderPath, new ShaderWithDateTime(shader, lastMod, false));
                return shader;
            }

            public static T AddCompiled(string shaderPath, string key, Func<T> compileFunc, Func<string, T> loadFunc)
            {
                var lastMod = File.GetLastWriteTimeUtc(shaderPath);

                ShaderWithDateTime result;
                if (s_CompiledShaders.TryGetValue(key, out result) &&
                    result.LastModified == lastMod)
                {
                    if (result.Shader != null)
                        return result.Shader;

                    try
                    {
                        if (loadFunc != null)
                        {
                            return loadFunc(result.CachePath);
                        }
                    }
                    catch
                    {
                        // Recompile if we encounter an error
                    }
                }

                if (result != null)
                {
                    DisposeHelper.Dispose(result.Shader);
                    File.Delete(result.CachePath);
                    s_CompiledShaders.Remove(key);
                    Modified = true;
                }

                T shader;
                try
                {
                    shader = compileFunc();
                }
                catch (SharpDX.CompilationException e)
                {
                    throw new SharpDX.CompilationException(e.ResultCode,
                        "Compilation Error in " + key + "\r\n\r\n" + e.Message);
                }

                s_CompiledShaders.Add(key, new ShaderWithDateTime(shader, lastMod, loadFunc != null));
                Modified = true;
                return shader;
            }

            [Serializable]
            private class ShaderWithDateTime
            {
                private readonly DateTime m_LastModified;
                private readonly string m_CachePath;

                [NonSerialized]
                private readonly T m_Shader;

                public T Shader
                {
                    get { return m_Shader; }
                }

                public DateTime LastModified
                {
                    get { return m_LastModified; }
                }

                public string CachePath
                {
                    get { return m_CachePath; }
                }

                public ShaderWithDateTime(T shader, DateTime lastModified, bool saveByteCode)
                {
                    m_Shader = shader;
                    m_LastModified = lastModified;
                    if (!saveByteCode)
                        return;

                    do
                    {
                        m_CachePath = Path.Combine(AppPath.GetUserDataDir("ShaderCache"),
                            string.Format("{0}.{1}", Guid.NewGuid(), "cso"));
                    } while (File.Exists(m_CachePath));
                    Directory.CreateDirectory(PathHelper.GetDirectoryName(m_CachePath));
                    File.WriteAllBytes(m_CachePath, shader.ObjectByteCode);
                }
            }

            public static void Save(string path)
            {
                using (var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Write))
                {
                    var bf = new BinaryFormatter();
                    bf.Serialize(fs, s_CompiledShaders);
                }
                Modified = false;
            }

            public static void Load(string path)
            {
                try
                {
                    using (var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
                    {
                        var bf = new BinaryFormatter();
                        s_CompiledShaders = (Dictionary<string, ShaderWithDateTime>) bf.Deserialize(fs);
                    }
                }
                catch
                {
                    // Ignore errors if we can't load
                }
                Modified = false;
            }
        }

        private static string ShaderCachePath
        {
            get { return Path.Combine(Cache<IShader>.ShaderCacheRoot, "ShaderIndex.dat"); }
        }

        private static string Shader11CachePath
        {
            get { return Path.Combine(Cache<IShader11>.ShaderCacheRoot, "Shader11Index.dat"); }
        }

        public static void Load()
        {
            if (s_Loaded)
                return;

            s_Loaded = true;
            Cache<IShader>.Load(ShaderCachePath);
            Cache<IShader11>.Load(Shader11CachePath);
        }

        private static string GetRelativePath(string rootPath, string filename)
        {
            if (!Path.IsPathRooted(filename))
                return filename;

            if (!filename.StartsWith(rootPath))
                throw new InvalidOperationException("No external shader files allowed");

            return filename.Remove(0, rootPath.Length + 1);
        }

        private static string GetRelative(string shaderFileName)
        {
            var asmPath = typeof (IRenderScript).Assembly.Location;
            var basePath = Path.Combine(PathHelper.GetDirectoryName(asmPath), "Extensions", "RenderScripts");
            return GetRelativePath(basePath, shaderFileName);
        }

        public static IShader CompileShader(string shaderFileName, string entryPoint = "main", string macroDefinitions = null)
        {
            var result = Cache<IShader>.AddCompiled(shaderFileName,
                String.Format("\"{0}\" /E {1} /D {2}", GetRelative(shaderFileName), entryPoint, macroDefinitions),
                () => Renderer.CompileShader(shaderFileName, entryPoint, macroDefinitions),
                Renderer.LoadShader);

            Cache<IShader>.Save(ShaderCachePath);
            return result;
        }

        public static IShader11 CompileShader11(string shaderFileName, string profile, string entryPoint = "main", string macroDefinitions = null)
        {
            var result = Cache<IShader11>.AddCompiled(shaderFileName,
                String.Format("\"{0}\" /E {1} /T {2} /D {3}", GetRelative(shaderFileName), entryPoint, profile, macroDefinitions),
                () => Renderer.CompileShader11(shaderFileName, entryPoint, profile, macroDefinitions),
                Renderer.LoadShader11);

            Cache<IShader11>.Save(ShaderCachePath);
            return result;
        }

        public static IKernel CompileClKernel(string sourceFileName, string entryPoint, string options = null)
        {
            return Cache<IKernel>.AddCompiled(sourceFileName,
                String.Format("\"{0}\" /E {1} /Opts {2}", GetRelative(sourceFileName), entryPoint, options),
                () => Renderer.CompileClKernel(sourceFileName, entryPoint, options),
                null);
        }

        public static IShader LoadShader(string shaderFileName)
        {
            return Cache<IShader>.AddLoaded(shaderFileName, Renderer.LoadShader);
        }

        public static IShader11 LoadShader11(string shaderFileName)
        {
            return Cache<IShader11>.AddLoaded(shaderFileName, Renderer.LoadShader11);
        }
    }
}