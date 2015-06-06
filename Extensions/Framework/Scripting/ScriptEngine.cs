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
//css_reference Microsoft.CSharp;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.Serialization;
using Microsoft.ClearScript;
using Microsoft.ClearScript.Windows;
using Mpdn.Extensions.Framework.RenderChain;
using Mpdn.Extensions.Framework.Scripting.ScriptEngineUtilities;
using Mpdn.RenderScript;
using ClearScriptEngine = Microsoft.ClearScript.ScriptEngine;

namespace Mpdn.Extensions.Framework.Scripting
{
    public class ScriptEngine : IDisposable
    {
        private ClearScriptEngine m_Engine;

        public HashSet<string> FilterTypeNames { get; private set; }
        public HashSet<string> EnumTypeNames { get; private set; }

        public ScriptEngine()
        {
            FilterTypeNames = new HashSet<string>();
            EnumTypeNames = new HashSet<string>();
            m_Engine = new JScriptEngine(WindowsScriptEngineFlags.EnableDebugging) {AllowReflection = true};
            m_Engine.AddHostType("Debug", typeof (Debug));

            AddEnumTypes(Assembly.GetAssembly(typeof (IRenderScript)));
            var asm = Assembly.GetExecutingAssembly();
            AddRenderScriptTypes(asm);
            AddEnumTypes(asm);
        }

        ~ScriptEngine()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        public virtual void Dispose(bool disposing)
        {
            if (!disposing) // ClearScript cannot be disposed from finalizer
                return;

            DisposeHelper.Dispose(ref m_Engine);
        }

        public IFilter Execute(RenderChain.RenderChain chain, IFilter input, string code, string filename = "")
        {
            try
            {
                var clip = ResetEngine(chain, input);
                m_Engine.Execute("RenderScript", true, code);
                return clip == null ? null : ((Clip) clip).Filter;
            }
            catch (ScriptEngineException e)
            {
                ThrowScriptEngineException(filename, e);
            }
            return null;
        }

        public bool Evaluate(RenderChain.RenderChain chain, IFilter input, string code, string filename = "")
        {
            try
            {
                ResetEngine(chain, input);
                dynamic result = m_Engine.Evaluate("RenderScript", true, code);
                return result is bool ? result : false;
            }
            catch (ScriptEngineException e)
            {
                ThrowScriptEngineException(filename, e);
            }
            return false;
        }

        private void ThrowScriptEngineException(string filename, ScriptEngineException e)
        {
            var message = m_Engine.GetStackTrace();
            throw new MpdnScriptEngineException(
                string.Format("Error in render script ('{0}'):\r\n\r\n{1}",
                    filename, string.IsNullOrEmpty(message) ? e.ErrorDetails : message));
        }

        private IClip ResetEngine(RenderChain.RenderChain chain, IFilter input)
        {
            m_Engine.CollectGarbage(true);
            var mock = (chain == null || input == null);
            var clip = mock ? (IClip) new MockClip() : new Clip(chain, input);
            AssignScriptObjects(clip);
            return mock ? null : clip;
        }

        private void AddEnumTypes(Assembly asm)
        {
            var enumTypes = asm.GetTypes().Where(t => t.IsEnum && t.IsPublic);
            foreach (var t in enumTypes)
            {
                if (EnumTypeNames.Contains(t.Name))
                {
                    throw new MpdnScriptEngineException(string.Format("Conflicting enum types detected: {0}", t.Name));
                }
                m_Engine.AddHostType(t.Name, t);
                EnumTypeNames.Add(t.Name);
            }
        }

        private void AddRenderScriptTypes(Assembly asm)
        {
            var filterTypes =
                asm.GetTypes()
                    .Where(
                        t =>
                            t.IsSubclassOf(typeof (RenderChain.RenderChain)) && t.IsPublic && !t.IsAbstract &&
                            t.GetConstructor(Type.EmptyTypes) != null);
            foreach (var t in filterTypes)
            {
                if (FilterTypeNames.Contains(t.Name))
                {
                    throw new MpdnScriptEngineException(string.Format("Conflicting render script types detected: {0}",
                        t.Name));
                }
                m_Engine.AddHostType(t.Name, t);
                FilterTypeNames.Add(t.Name);
            }
        }

        private void AssignScriptObjects(IClip clip)
        {
            m_Engine.Script["input"] = clip;
            m_Engine.Script["Script"] = new Script();
            m_Engine.Script["Gpu"] = Renderer.Dx9GpuInfo.Details;
            m_Engine.Script["__$xhost"] = new InternalHostFunctions();
            m_Engine.Script["Host"] = new Host();
        }
    }

    public class MpdnScriptEngineException : Exception
    {
        public MpdnScriptEngineException()
        {
        }

        public MpdnScriptEngineException(string message) : base(message)
        {
        }

        public MpdnScriptEngineException(string message, Exception innerException) : base(message, innerException)
        {
        }

        protected MpdnScriptEngineException(SerializationInfo info, StreamingContext context) : base(info, context)
        {
        }
    }
}
