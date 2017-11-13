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
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;

namespace Mpdn.Extensions.Framework
{
    public interface ITagged
    {
        IProcessData ProcessData { get; }
    }

    #region Mergeable Data structures

    public interface IMergeable<in T>
    {
        void MergeWith(T data);
    }

    public interface IUnionFind<TData> : IMergeable<IUnionFind<TData>>, IMergeable<TData>
    {
        TData Data { get; }

        int Depth { get; set; }
        IUnionFind<TData> Root { get; set; }
    }

    public static class MergeableHelper
    {
        public static void Add<T>(this IMergeable<IEnumerable<T>> mergeable, T data)
        {
            mergeable.MergeWith(new[] { data });
        }
    }

    public class MergeableCollection<TValue> : Collection<TValue>, IMergeable<IEnumerable<TValue>>
    {
        public void MergeWith(IEnumerable<TValue> data)
        {
            foreach (var x in data) Add(x);
        }
    }

    public class UnionFind<IData, TData> : IUnionFind<IData>
        where TData : class, IData, IMergeable<IData>, new()
    {
        // Union-find structure augmented with data

        #region IUnionFind Implementation
       
        public IData Data { get { return (m_Root == this) ? (m_Data ?? (m_Data = new TData())) : Root.Data; } }

        public int Depth { get; set; }

        public IUnionFind<IData> Root
        {
            get
            {
                if (m_Root != this)
                    m_Root = m_Root.Root;
                return m_Root;
            }

            set
            {
                if (value != m_Root && m_Data != null)
                {
                    value.MergeWith(m_Data);
                    m_Data = null;
                }
                m_Root = value;
            }
        }

        public void MergeWith(IUnionFind<IData> tree)
        {
            MergeRoots(Root, tree.Root);
        }

        public void MergeWith(IData data)
        {
            if (Root != this)
                Root.MergeWith(data);
            else
                m_Data.MergeWith(data);
        }

        #endregion

        #region Implementation

        private IUnionFind<IData> m_Root;

        private TData m_Data;

        private static void MergeRoots(IUnionFind<IData> x, IUnionFind<IData> y)
        {
            if (x == y)
                return;
            if (x.Depth < y.Depth)
                x.Root = y;
            else if (x.Depth > y.Depth)
                y.Root = x;
            else
            {
                y.Root = x;
                x.Depth += 1;
            }
        }

        public UnionFind() { m_Root = this; }

        #endregion
    }

    public class UnionCollection<TValue> : UnionFind<IEnumerable<TValue>, MergeableCollection<TValue>> { }

    #endregion

    public struct ProcessInterval
    {
        public readonly int Min;
        public readonly int Max;

        public ProcessInterval(int low, int high)
        {
            Min = low;
            Max = high;
        }

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

    public interface IProcessData : IProcessRange, IUnionFind<IEnumerable<IProcessTag>>
    {
        int Rank { set; }
        void AddInput(IProcessData process);
    }

    public class ProcessRange : IProcessRange
    {
        public ProcessRange(IProcessRange start, IProcessRange end)
        {
            m_Start = start;
            m_End = end ?? start;

            if (m_End == null)
                throw new ArgumentNullException("Either 'start' or 'end' needs to be non-null.");
        }

        #region IProcessRange Implementation

        private IProcessRange m_Start;
        private IProcessRange m_End;

        public ProcessInterval Range(IDictionary<IProcessRange, ProcessInterval> cache)
        {
            return new ProcessInterval(
                (m_Start != null)
                    ? m_Start.Range(cache).Max
                    : m_End.Range(cache).Min,
                m_End.Range(cache).Max);
        }

        #endregion
    }

    public abstract class ProcessTagBase : IProcessTag
    {
        public abstract string Label { get; }

        public abstract int Verbosity { get; }

        public ProcessTagBase(IProcessRange range)
        {
            m_Range = range;
        }

        public override string ToString() { return Label; }

        #region IProcessRange Implementation

        private IProcessRange m_Range;

        public ProcessInterval Range(IDictionary<IProcessRange, ProcessInterval> cache)
        {
            return m_Range.Range(cache);
        }

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
        public int Rank { private get; set; }

        #region Implementation

        private IList<IProcessData> m_InputProcesses = new List<IProcessData>();

        public ProcessInterval Range(IDictionary<IProcessRange, ProcessInterval> cache)
        {
            if (Rank > 0)
                return new ProcessInterval(Rank, Rank+1);

            ProcessInterval interval;
            if (cache.TryGetValue(this, out interval))
                return interval;

            int max = -1;
            foreach (var process in m_InputProcesses)
                max = Math.Max(max, process.Range(cache).Max);

            return (cache[this] = new ProcessInterval(max, max));
        }

        public void AddInput(IProcessData process)
        {
            if (process != this)
            {
                m_InputProcesses.Add(process);
                MergeWith(process);
            }
        }

        public ProcessData() { Rank = -1; }

        #endregion
    }
    
    public static class TagHelper
    {
        public static void AddTag(this ITagged tagged, IProcessTag tag)
        {
            //tagged.ProcessData.Add(tag);
        }

        public static void AddLabel(this ITagged tagged, string label, int verbosity = 0, ITagged start = null)
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
    
        private const int DefaultVerbosity = 0;

        public static string CreateString(this IProcessData process, int verbosity = DefaultVerbosity)
        {
            var tags = process.Data.Where(t => (t.Verbosity <= verbosity)).ToList();

            var rangecache = new Dictionary<IProcessRange, ProcessInterval>();
            var range = tags.ToDictionary(
                tag => tag,
                tag => tag.Range(rangecache));

            tags = (from tag in tags
                    where range[tag].NonEmpty()
                    orderby range[tag].Max descending
                    select tag)
                    .ToList();

            var depth = tags.ToDictionary(
                tag => tag,
                tag => tags
                    .TakeWhile(t => range[t].Max > range[tag].Max)
                    .Count(t => range[t].Contains(range[tag])));
            tags.Reverse();

            return string.Join("\n",
                from tag in tags
                select String.Concat(Enumerable.Repeat("    ", depth[tag])) + tag.Label);
        }

        #endregion
    }
}