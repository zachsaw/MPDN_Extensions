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

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Shiandow.Merging;

namespace Mpdn.Extensions.Framework
{
    public interface ITagged
    {
        IProcessData ProcessData { get; }
    }

    public struct ProcessInterval
    {
        public readonly int Min;
        public readonly int Max;

        public ProcessInterval(int low, int high)
        {
            Min = low;
            Max = high;
        }

        #region Helper Methods

        public bool Contains(ProcessInterval x)
        {
            return (Min <= x.Min) && (x.Max <= Max);
        }

        public bool NonEmpty()
        {
            return Min < Max;
        }

        public override string ToString()
        {
            return String.Format("({0}, {1}]", Min, Max);
        }

        #endregion
    }

    public interface IProcessData : IProcessRange, IUnionFind<IEnumerable<IProcessTag>>
    {
        void AddInput(IProcessData process);
        IEnumerable<IProcessData> Enumerate(ISet<IProcessData> visited);
    }

    public interface IProcessRange
    {
        ProcessInterval Range(IDictionary<IProcessRange, ProcessInterval> cache);
    }

    public interface IProcessTag : IProcessRange 
    {
        string Label { get; }
        int Verbosity { get; }
    }

    public class ProcessRange : IProcessRange
    {
        public ProcessRange(IProcessRange start, IProcessRange end)
        {
            if (start == null)
                throw new ArgumentNullException("start");
            if (end == null)
                throw new ArgumentNullException("end");

            m_Start = start;
            m_End = end;
        }

        #region IProcessRange Implementation

        private IProcessRange m_Start;
        private IProcessRange m_End;

        public ProcessInterval Range(IDictionary<IProcessRange, ProcessInterval> cache)
        {
            ProcessInterval interval = cache.TryGetValue(this, out interval)
                ? interval
                : cache[this] = new ProcessInterval(m_Start.Range(cache).Max, m_End.Range(cache).Max);
            return interval;
        }

        #endregion
    }

    public abstract class ProcessTagBase : IProcessTag
    {
        public abstract string Label { get; }

        public abstract int Verbosity { get; }

        #region IProcessRange Implementation

        private readonly IProcessRange m_Range;

        public ProcessTagBase(IProcessRange range) { m_Range = range; }

        public ProcessInterval Range(IDictionary<IProcessRange, ProcessInterval> cache) { return m_Range.Range(cache); } 

        public override string ToString() { return Label; }

        #endregion
    }

    public class ProcessTag : ProcessTagBase
    {
        public override string Label { get { return m_Label; } }

        public override int Verbosity { get { return m_Verbosity; } }

        #region Implementation

        private readonly string m_Label;
        private readonly int m_Verbosity;

        public ProcessTag(string label, int verbostiy, IProcessRange range)
            : base(range)
        {
            m_Label = label;
            m_Verbosity = verbostiy;
        }

        #endregion
    }

    public class DeferredTag : ProcessTagBase
    {
        public override string Label { get { return m_Func(); } }

        public override int Verbosity { get { return m_VerbosityFunc(); } }

        #region Implementation

        private readonly Func<string> m_Func;
        private readonly Func<int> m_VerbosityFunc;

        public DeferredTag(Func<string> func, IProcessRange range)
            : this(func, () => String.IsNullOrEmpty(func()) ? 1000 : 1, range)
        { }

        public DeferredTag(string label, Func<int> verbosityFunc, IProcessRange range)
            : this(() => label, verbosityFunc, range)
        { }

        public DeferredTag(Func<string> func, Func<int> verbosityFunc, IProcessRange range)
            : base(range)
        {
            m_Func = func;
            m_VerbosityFunc = verbosityFunc;
        }

        #endregion
    }

    public class ProcessData : UnionCollection<IProcessTag>, IProcessData
    {
#if DEBUG
        public string DebugDescription { get { return this.CreateString(10); } }
#endif

        #region Implementation

        private readonly IList<IProcessData> m_InputProcesses = new List<IProcessData>();

        public void AddInput(IProcessData process)
        {
            if (process != this)
            {
                m_InputProcesses.Add(process);
                MergeWith(process);
            }
        }

        public ProcessInterval Range(IDictionary<IProcessRange, ProcessInterval> cache)
        {
            ProcessInterval interval;
            if (cache.TryGetValue(this, out interval))
                return interval;
            
            int rank = m_InputProcesses
                .Select(x => x.Range(cache).Max)
                .Concat(new[] { -1 })
                .Max();
            return cache[this] = new ProcessInterval(rank, rank);
        }

        public IEnumerable<IProcessData> Enumerate(ISet<IProcessData> visited)
        {
            if (visited.Contains(this))
                yield break;

            visited.Add(this);
            foreach (var child in m_InputProcesses.SelectMany(p => p.Enumerate(visited)))
                yield return child;
            yield return this;
        }

        #endregion
    }
    
    public static class TagHelper
    {
        public static void AddTag(this ITagged tagged, IProcessTag tag)
        {
            tagged.ProcessData.Add(tag);
        }

        public static void AddLabel(this ITagged tagged, string label, int verbosity = 1, ITagged start = null)
        {
            tagged.AddTag(new ProcessTag(label, String.IsNullOrEmpty(label) ? 1000 : verbosity, 
                (start == null) 
                ? (IProcessRange)tagged.ProcessData
                : new ProcessRange(start.ProcessData, tagged.ProcessData)));
        }

        public static TTagged Tagged<TTagged>(this TTagged tagged, IProcessTag tag)
            where TTagged : ITagged
        {
            tagged.AddTag(tag);
            return tagged;
        }

        public static TTagged Labeled<TTagged>(this TTagged taggedProcess, string label, int verbosity = 0, ITagged start = null)
            where TTagged : ITagged
        {
            taggedProcess.AddLabel(label, verbosity, start);
            return taggedProcess;
        }

        #region String Generation
    
        private const int DefaultVerbosity = 1;

        public static string CreateString(this IProcessData process, int verbosity = DefaultVerbosity)
        {
            var processes = process.Enumerate(new HashSet<IProcessData>()).ToList();
            var tags = process.Data.Where(t => (t.Verbosity <= verbosity)).ToList();

            var cache = processes
                .Select((p, k) => new { Key = (IProcessRange)p, Value = new ProcessInterval(k, k + 1) })
                .ToDictionary(x => x.Key, x => x.Value);
            var range = tags.ToDictionary(tag => tag, tag => tag.Range(cache));

            tags = (from tag in tags
                    where range[tag].NonEmpty()
                    orderby range[tag].Max descending
                    select tag)
                    .ToList();

            var ranges = new List<ProcessInterval>();
            var depth = tags.ToDictionary(
                tag => tag,
                tag => (ranges = ranges
                    .Where(r => r.Min <= range[tag].Min)
                    .Concat(new[] { range[tag] })
                    .ToList())
                    .Count() - 1);
            tags.Reverse();

            return string.Join("\n",
                from tag in tags
                select String.Concat(Enumerable.Repeat("    ", depth[tag])) + tag.Label);
        }

        #endregion
    }
}