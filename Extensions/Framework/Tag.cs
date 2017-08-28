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

namespace Mpdn.Extensions.Framework
{
    public interface ITaggedProcess
    {
        IProcessData ProcessData { get; }
    }

    public interface IUnionTree<TData> : IEnumerable<TData>
    {
        ISet<TData> Set { get; }

        int Depth { get; set; }
        IUnionTree<TData> Root { get; set; }

        void MergeWith(IUnionTree<TData> tree);
    }

    public interface IProcessData
    {
        IUnionTree<ProcessTag> TagTree { get; }
        ISet<ProcessTag> Tags { get; } // => TagTree.Set

        int InputRank { get; }
        int Rank { get; set; }

        void AddInputs(IEnumerable<IProcessData> inputProcesses);
    }

    public class UnionTree<TData> : IUnionTree<TData>
    {
        public ISet<TData> Set { get { return (m_Root == this) ? (m_Set ?? (m_Set = new HashSet<TData>())) : Root.Set; } }

        public int Depth { get; set; }

        public IUnionTree<TData> Root
        {
            get
            {
                if (m_Root != this)
                    m_Root = m_Root.Root;
                return m_Root;
            }

            set
            {
                if (value != m_Root && m_Set != null)
                {
                    value.Set.UnionWith(m_Set);
                    m_Set = null;
                }
                m_Root = value;
            }
        }

        public void MergeWith(IUnionTree<TData> tree)
        {
            MergeRoots(Root, tree.Root);
        }

        #region Implementation

        private IUnionTree<TData> m_Root;

        private ISet<TData> m_Set;

        private static void MergeRoots(IUnionTree<TData> x, IUnionTree<TData> y)
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

        public IEnumerator<TData> GetEnumerator()
        {
            return Set.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public UnionTree()
        {
            m_Root = this;
        }

        #endregion
    }

    public struct ProcessTag
    {
        public string Label { get; private set; }

        public int Verbosity { get; private set; }

        public Tuple<int, int> Range()
        {
            if (m_Start != null)
                return new Tuple<int, int>(m_Start.Rank, m_End.Rank);
            else
                return new Tuple<int, int>(m_End.InputRank, m_End.Rank);
        }

        #region Implementation

        private IProcessData m_Start;

        private IProcessData m_End;

        public override string ToString()
        {
            return Label;
        }

        public ProcessTag(string label, int verbostiy, IProcessData start, IProcessData end)
        {
            Label = label;
            Verbosity = verbostiy;
            m_Start = start;
            m_End = end ?? start;

            if (m_End == null)
                throw new ArgumentNullException("Either 'start' or 'end' needs to be non-null.");
        }

        #endregion
    }

    public class ProcessData : IProcessData
    {
        public IUnionTree<ProcessTag> TagTree { get; private set; }

        public ISet<ProcessTag> Tags { get { return TagTree.Set; } }

        public int InputRank
        {
            get
            {
                if (m_InputRank < 0)
                    foreach (var process in m_InputProcesses)
                        m_InputRank = Math.Max(m_InputRank, process.Rank);
                return m_InputRank;
            }
        }

        public int Rank
        {
            get
            {
                if (m_Rank < 0)
                    m_Rank = InputRank;
                return m_Rank;
            }

            set { m_Rank = value; }
        }

        #region Implementation

        private int m_InputRank = -1;

        private int m_Rank = -1;

        private IList<IProcessData> m_InputProcesses = new List<IProcessData>();

        private void MergeWith(IProcessData process)
        {
            TagTree.MergeWith(process.TagTree);
        }

        public void AddInputs(IEnumerable<IProcessData> inputProcesses)
        {
            foreach (var process in inputProcesses)
            if (process != this)
            {
                m_InputProcesses.Add(process);
                MergeWith(process);
            }
        }

        public ProcessData()
        {
            TagTree = new UnionTree<ProcessTag>();
        }

        #endregion
    }
    
    public static class TagHelper
    {
        public static void AddTag(this IProcessData processData, ProcessTag tag)
        {
            processData.Tags.Add(tag);
        }

        public static void AddTag(this ITaggedProcess taggedProcess, ProcessTag tag)
        {
            taggedProcess.ProcessData.AddTag(tag);
        }

        public static void AddLabel(this ITaggedProcess taggedProcess, string label, int verbosity = 0, ITaggedProcess start = null)
        {
            taggedProcess.AddTag(new ProcessTag(label, String.IsNullOrEmpty(label) ? 1000 : verbosity, start != null ? start.ProcessData : null, taggedProcess.ProcessData));
        }

        public static TtaggedProcess Tagged<TtaggedProcess>(this TtaggedProcess taggedProcess, ProcessTag tag)
            where TtaggedProcess : ITaggedProcess
        {
            taggedProcess.AddTag(tag);
            return taggedProcess;
        }

        public static TtaggedProcess Labeled<TtaggedProcess>(this TtaggedProcess taggedProcess, string label, int verbosity = 0, ITaggedProcess start = null)
            where TtaggedProcess : ITaggedProcess
        {
            taggedProcess.AddLabel(label, verbosity, start);
            return taggedProcess;
        }

        #region String Generation

        private const int DefaultVerbosity = 0;

        public static string CreateString(this IProcessData data, int verbosity = DefaultVerbosity)
        {
            var tags = data.Tags.Where(t => (t.Verbosity <= verbosity)).ToList();

            var range = tags.ToDictionary(
                tag => tag,
                tag => tag.Range());

            tags = tags.Where(t => (range[t].Item1 < range[t].Item2)).ToList();

            var depth = tags.ToDictionary(
                tag => tag,
                tag => tags.Count(t => 
                        range[t].Item1 <= range[tag].Item1 && range[tag].Item2 <= range[t].Item2
                    && (range[t].Item1 != range[tag].Item1 || range[tag].Item2 != range[t].Item2)));

            var labels = tags
                .Where(n => -n.Verbosity <= verbosity)
                .Select(n => String.Concat(Enumerable.Repeat("    ", depth[n])) + n.Label);
            return string.Join("\n", labels);
        }

        #endregion
    }
}