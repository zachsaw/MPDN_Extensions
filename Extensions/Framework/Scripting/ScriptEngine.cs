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
using Microsoft.ClearScript;
using Microsoft.ClearScript.Windows;
using Mpdn.Extensions.Framework.Exceptions;
using Mpdn.Extensions.Framework.RenderChain;
using Mpdn.Extensions.Framework.Scripting.ScriptEngineUtilities;
using Mpdn.RenderScript;
using ClearScriptEngine = Microsoft.ClearScript.ScriptEngine;

namespace Mpdn.Extensions.Framework.Scripting
{
    public abstract class ScriptEngine<TChain, TInput, TOutput> : IDisposable
        where TChain : class
    {
        protected ClearScriptEngine Engine;

        public HashSet<string> FilterTypeNames { get; private set; }
        public HashSet<string> EnumTypeNames { get; private set; }

        protected ScriptEngine()
        {
            FilterTypeNames = new HashSet<string>();
            EnumTypeNames = new HashSet<string>();
            Engine = new JScriptEngine(WindowsScriptEngineFlags.EnableDebugging) {AllowReflection = true};
            Engine.AddHostType("Debug", typeof (Debug));

            AddEnumTypes(Assembly.GetAssembly(typeof (IScript)));
            foreach (var asm in (Extension.Scripts.Select(s => s.GetType().Assembly)
                .Concat(new[] {typeof (TChain).Assembly})).Distinct())
            {
                AddScriptTypes(asm);
                AddEnumTypes(asm);
            }
        }

        public TOutput Execute(TInput input, string code, string filename = "")
        {
            try
            {
                var clip = ResetEngine(input);
                Engine.Execute("Script", true, code);
                return clip;
            }
            catch (ScriptEngineException e)
            {
                ThrowScriptEngineException(filename, e);
            }
            return default(TOutput);
        }

        public bool Evaluate(TInput input, string code, string filename = "")
        {
            try
            {
                ResetEngine(input);
                dynamic result = Engine.Evaluate("Script", true, code);
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
            var message = Engine.GetStackTrace();
            throw new MpdnScriptEngineException(
                string.Format("Error in script ('{0}'):\r\n\r\n{1}",
                    filename, string.IsNullOrEmpty(message) ? e.ErrorDetails : message));
        }

        protected abstract TOutput Reset(TInput input);

        private TOutput ResetEngine(TInput input)
        {
            Engine.CollectGarbage(true);
            return Reset(input);
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
                Engine.AddHostType(t.Name, t);
                EnumTypeNames.Add(t.Name);
            }
        }

        private void AddScriptTypes(Assembly asm)
        {
            var filterTypes = asm.GetTypes()
                .Where(t =>
                    t.IsSubclassOf(typeof (TChain)) && t.IsPublic && !t.IsAbstract &&
                    t.GetConstructor(Type.EmptyTypes) != null);
            foreach (var t in filterTypes)
            {
                if (FilterTypeNames.Contains(t.Name))
                {
                    throw new MpdnScriptEngineException(string.Format("Conflicting script types detected: {0}",
                        t.Name));
                }
                Engine.AddHostType(t.Name, t);
                FilterTypeNames.Add(t.Name);
            }
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

        protected virtual void Dispose(bool disposing)
        {
            if (!disposing)
                return;

            DisposeHelper.Dispose(ref Engine);
        }
    }

    public class RenderScriptEngine : ScriptEngine<RenderChain.RenderChain, IFilter, Clip>
    {
        private static RenderScriptEngine s_ScriptEngine;

        private static RenderScriptEngine Instance
        {
            get
            {
                if (s_ScriptEngine != null) return s_ScriptEngine;
                s_ScriptEngine = new RenderScriptEngine();
                Player.Closed += (sender, args) => s_ScriptEngine.Dispose();
                return s_ScriptEngine;
            }
        }

        public static FilterClip Exec(IFilter input, string code, string filename = "")
        {
            return (FilterClip) Instance.Execute(input, code, filename);
        }

        public static bool Eval(IFilter input, string code, string filename = "")
        {
            return Instance.Evaluate(input, code, filename);
        }

        public static HashSet<string> FilterTypes { get { return Instance.FilterTypeNames; } }
        public static HashSet<string> EnumTypes { get { return Instance.EnumTypeNames; } }

        protected override Clip Reset(IFilter input)
        {
            var mock = input == null;
            var clip = mock ? (Clip) new MockFilterClip() : new FilterClip(input);
            AssignScriptObjects(clip);
            return mock ? null : clip;
        }

        private void AssignScriptObjects(Clip clip)
        {
            Engine.Script["input"] = clip;
            Engine.Script["Script"] = new Script();
            Engine.Script["Gpu"] = Renderer.Dx9GpuInfo.Details;
            Engine.Script["__$xhost"] = new InternalHostFunctions();
            Engine.Script["Host"] = new Host();
        }
    }
}
