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
using System.Linq;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Threading.Tasks;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using Mpdn.OpenCl;
using Mpdn.RenderScript;
using SharpDX;
using System.Threading;

namespace Mpdn.Extensions.Framework.RenderChain
{
    public static class ShaderCache
    {
        public static string ShaderPathRoot
        {
            get
            {
                var asmPath = typeof(IRenderScript).Assembly.Location;
#if DEBUG
                return Path.GetFullPath(Path.Combine(PathHelper.GetDirectoryName(asmPath), "..\\", "Extensions", "RenderScripts"));
#else
                return Path.Combine(PathHelper.GetDirectoryName(asmPath), "Extensions", "RenderScripts");
#endif
            }
        }

        public static string ShaderCacheRoot
        {
            get { return AppPath.GetUserDataDir("ShaderCache"); }
        }

        public static void Load()
        {
            if (s_Init.Value)
                return;

            throw new InvalidOperationException("Shader cache loading failed.");
        }

        public static Task Prefetch()
        {
            return s_InitAsync.Value;
        }

        public static string GetRelativePath(string rootPath, string filename)
        {
            if (!Path.IsPathRooted(filename))
                return filename;

            if (!filename.StartsWith(rootPath))
                return filename;

            return filename.Remove(0, rootPath.Length + 1);
        }

        private static string GetRelative(string shaderFileName)
        {
            return GetRelativePath(ShaderPathRoot, Path.GetFullPath(shaderFileName));
        }

        public static IShader CompileShader(string shaderFileName, string profile = "ps_3_0", string entryPoint = "main", string macroDefinitions = null)
        {
            var result = Cache<IShader>.AddCompiled(shaderFileName,
                string.Format("\"{0}\" /E {1} /T {2} /D {3}", GetRelative(shaderFileName), entryPoint, profile, macroDefinitions),
                new Lazy<IShader>(() => Renderer.CompileShader(shaderFileName, entryPoint, profile, macroDefinitions)),
                Renderer.LoadShader);

            return result;
        }

        public static IShader11 CompileShader11(string shaderFileName, string profile, string entryPoint = "main", string macroDefinitions = null)
        {
            var result = Cache<IShader11>.AddCompiled(shaderFileName,
                string.Format("\"{0}\" /E {1} /T {2} /D {3}", GetRelative(shaderFileName), entryPoint, profile, macroDefinitions),
                new Lazy<IShader11>(() => Renderer.CompileShader11(shaderFileName, entryPoint, profile, macroDefinitions)),
                Renderer.LoadShader11);

            return result;
        }

        public static IKernel CompileClKernel(string sourceFileName, string entryPoint, string options = null)
        {
            return Cache<IKernel>.AddCompiled(sourceFileName,
                string.Format("\"{0}\" /E {1} /Opts {2}", GetRelative(sourceFileName), entryPoint, options),
                new Lazy<IKernel>(() => Renderer.CompileClKernel(sourceFileName, entryPoint, options)),
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

        #region Implementation

        private static Lazy<Task<bool>> s_InitAsync = new Lazy<Task<bool>>(() => Task.Run(() => s_Init.Value));
        private static Lazy<bool> s_Init = new Lazy<bool>(() =>
        {
            PlayerControl.FormClosed += (o, e) =>
            {
                TrimCache();
            };

            return Cache<IShader>.Initialized && Cache<IShader11>.Initialized && Cache<IKernel>.Initialized;
        });

        private static void TrimCache()
        {
            var s = new CancellationTokenSource();
            var token = s.Token;
            Task.Run(() =>
            {
                var files = Directory
                    .EnumerateFiles(ShaderCacheRoot)
                    .Except(s_CachedFiles.SelectMany(enumeration => enumeration()));

                foreach (var orphan in files)
                {
                    if (token.IsCancellationRequested)
                        break;

                    File.Delete(orphan);
                }
            }, token);

            s.CancelAfter(1000); // Ensure cleaning doesn't take ages.
        }

        private static IList<Func<IEnumerable<string>>> s_CachedFiles = new List<Func<IEnumerable<string>>>();

        private static class Cache<T> where T : class, IShaderBase
        {
            private static ConcurrentDictionary<string, ShaderWithDateTime> s_CompiledShaders;

            public static bool Initialized { get { return s_Initialized.Value; } }

            public static T AddLoaded(string shaderPath, Func<string, T> loadFunc)
            {
                return AddCompiled(shaderPath, shaderPath, new Lazy<T>(() => loadFunc(shaderPath)), null);
            }

            public static T AddCompiled(string shaderPath, string key, Lazy<T> compileFunc, Func<string, T> loadFunc)
            {
                return AddOrUpdate(shaderPath, key, compileFunc, loadFunc);
            }

            #region Implementation

            private static volatile bool s_Saved;

            private static Lazy<bool> s_Initialized = new Lazy<bool>(() =>
            {
                Load(CacheFilePath);

                s_CachedFiles.Add(
                    () => s_CompiledShaders
                    .ToArray()
                    .Select(x => x.Value.CachePath)
                    .Concat(new[] { CacheFilePath }));

                PlayerControl.FormClosed += (o, e) =>
                {
                    Save(CacheFilePath);
                };
                return true;
            });

            private static string CacheFilePath
            {
                get
                {
                    var fileName = typeof(T).Name.TrimStart('I') + "Index.dat";
                    return Path.Combine(ShaderCacheRoot, fileName);
                }
            }

            private static void Save(string path)
            {
                if (s_Saved)
                    return;

                using (var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Write))
                {
                    var bf = new BinaryFormatter();
                    bf.Serialize(fs, new Dictionary<string, ShaderWithDateTime>(s_CompiledShaders));
                }
                s_Saved = true;
            }

            private static void Load(string path)
            {
                try
                {
                    using (var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
                    {
                        var bf = new BinaryFormatter();
                        s_CompiledShaders = new ConcurrentDictionary<string, ShaderWithDateTime>((Dictionary<string, ShaderWithDateTime>)bf.Deserialize(fs));
                    }
                    s_Saved = true; // Everything already on disk
                }
                catch
                {
                    s_CompiledShaders = new ConcurrentDictionary<string, ShaderWithDateTime>();// Ignore errors if we can't load
                }
            }

            private static Lazy<T> SafeCompile(string key, Lazy<T> compileFunc)
            {
                return new Lazy<T>(() =>
                {
                    s_Saved = false;

                    T shader = null;
                    try
                    {
                        shader = compileFunc.Value;
                        return shader;
                    }
                    catch (CompilationException e)
                    {
                        throw new CompilationException(e.ResultCode, "Compilation Error in " + key + "\r\n\r\n" + e.Message);
                    }
                    catch (OpenClException e)
                    {
                        throw new OpenClException("Compilation Error in " + key + "\r\n\r\n" + e.Message, e.ErrorCode);
                    }
                });
            }

            private static T AddOrUpdate(string shaderPath, string key, Lazy<T> compileFunc, Func<string, T> loadFunc)
            {
                if (!Initialized)
                    return null;

                var lastMod = File.GetLastWriteTimeUtc(shaderPath);

                var safeShader = SafeCompile(key, compileFunc);

                Func<string, ShaderWithDateTime> addEntry =
                    (_) => new ShaderWithDateTime(safeShader.Value, lastMod, loadFunc != null);

                Func<string, ShaderWithDateTime, ShaderWithDateTime> updateEntry = (_, previous) =>
                {
                    if (previous.LastModified == lastMod)
                    {
                        if (previous.Shader == null && loadFunc != null)
                        {
                            previous.LoadWith(loadFunc);
                        }

                        if (previous.Shader != null)
                            return previous;
                    }

                    DisposeHelper.Dispose(previous);

                    return new ShaderWithDateTime(safeShader.Value, lastMod, loadFunc != null);
                };

                var entry = s_CompiledShaders.AddOrUpdate(key, addEntry, updateEntry);
                return entry.Shader;
            }

            [Serializable]
            private class ShaderWithDateTime : IDisposable
            {
                private readonly DateTime m_LastModified;
                private readonly string m_CachePath;

                [NonSerialized]
                private T m_Shader;

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

                protected ShaderWithDateTime() { }

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

                public void Dispose()
                {
                    if (CachePath != null)
                        File.Delete(CachePath);
                    DisposeHelper.Dispose(Shader);
                }

                public void LoadWith(Func<string, T> loadFunc)
                {
                    if (CachePath != null)
                        try
                        {
                            Interlocked.CompareExchange(ref m_Shader, loadFunc(CachePath), null);
                        }
                        catch { /* Ignore Errors */ }
                }
            }

            #endregion
        }

        #endregion
    }
}