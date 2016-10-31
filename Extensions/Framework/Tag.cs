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
using System.Linq;

namespace Mpdn.Extensions.Framework
{
    public interface ITagged
    {
        ProcessTag Tag { get; }
    }

    public abstract class ProcessTag
    {
        public static readonly ProcessTag BOTTOM = new BottomTag();

        public abstract string Label { get; }

        public abstract int Verbosity { get; }

        #region Base Implementation

        private readonly HashSet<ProcessTag> m_InputTags = new HashSet<ProcessTag>();

        protected virtual ISet<ProcessTag> InputTags
        {
            get { return m_InputTags; }
        }

        #endregion

        #region Text rendering

#if DEBUG
        protected const int DefaultVerbosity = 0;
        protected string TreeDescription { get { return CreateString(100); } }
#else
        protected const int DefaultVerbosity = 0;
#endif

        public string CreateString(int verbosity = DefaultVerbosity)
        {
            var subNodes = SubGraph(n => n.Verbosity <= verbosity);
            var nodes = subNodes.Keys;

            var index = subNodes
                .Select((n, i) => new KeyValuePair<ProcessTag, int>(n.Key, i))
                .ToDictionary(x => x.Key, x => x.Value);

            var minIndex = subNodes.ToDictionary(x => x.Key, 
                x => x.Value.Select(n => index[n])
                    .Aggregate(index[x.Key], Math.Min));

            var depth = nodes.ToDictionary(
                node => node, 
                node => nodes.Count(n => minIndex[n] < index[node] && index[node] < index[n]));

            var labels = nodes
                .Where(n => -n.Verbosity <= verbosity)
                .Select(n => String.Concat(Enumerable.Repeat("    ", depth[n])) + n.Label);
            return string.Join("\n", labels);
        }

        #endregion

        #region Graph Operations

        protected void AddInput(ProcessTag tag)
        {
            InputTags.Add(tag);
        }

        protected void AddInputs(IEnumerable<ProcessTag> tags)
        {
            foreach (var tag in tags)
                AddInput(tag);
        }

        protected void RemoveInput(ProcessTag tag)
        {
            m_InputTags.Remove(tag);
        }

        protected void RemoveInputs(IEnumerable<ProcessTag> tags)
        {
            foreach (var tag in tags)
                RemoveInput(tag);
        }

        public void AddPrefix(ProcessTag prefix)
        {
            foreach (var tag in EndNodes())
                tag.AddInput(prefix);
        }

        public void Remove(ProcessTag tag)
        {
            if (InputTags.Contains(tag))
            {
                InputTags.ExceptWith(new []{ tag });
                AddInputs(tag.InputTags.Except(new [] { tag }));
            }
        }

        public void Purge(ProcessTag tag)
        {
            foreach (var node in Nodes())
                node.Remove(tag);
        }

        public void Insert(ProcessTag tag)
        {
            tag.AddPrefix(new HubTag(InputTags.ToArray()));
            InputTags.Clear();
            InputTags.Add(tag);
        }

        public bool IsEndNode()
        {
            return !InputTags.Any();
        }

        public bool HasAncestor(ProcessTag ancestor)
        {
            return ancestor.TraverseTree().Contains(this);
        }

        public bool ConnectedTo(ProcessTag tag)
        {
            return tag.HasAncestor(this) || this.HasAncestor(tag);
        }

        public virtual ISet<ProcessTag> SubGraphNodes(Func<ProcessTag, bool> predicate)
        {
            return FilteredGraphInputNodes(predicate);
        }

        protected virtual IDictionary<ProcessTag, ISet<ProcessTag>> SubGraph(Func<ProcessTag, bool> predicate)
        {
            return FilteredGraph(predicate);
        }

        protected ISet<ProcessTag> FilteredGraphInputNodes(Func<ProcessTag, bool> predicate)
        {
            var nodes = new HashSet<ProcessTag>();
            var visited = new HashSet<ProcessTag>();
            var ignoredNodes = TraverseIf((tag) =>
            {
                if (tag != this && predicate(tag))
                {
                    nodes.Add(tag);
                    return false;
                }

                if (visited.Contains(tag))
                    return false;

                visited.Add(tag);
                return true;
            }).ToList();
            return nodes;
        }

        private IDictionary<ProcessTag, ISet<ProcessTag>> FilteredGraph(Func<ProcessTag, bool> predicate)
        {
            return TraverseTree()
                .Where(predicate)
                .ToDictionary(node => node, node => node.SubGraphNodes(predicate)); ;
        }

        #endregion

        #region Node Enumeration

        private IEnumerable<ProcessTag> TraverseIf(Func<ProcessTag, bool> predicate)
        {
            if (!predicate(this))
                yield break;

            foreach (var tag in InputTags)
                foreach (var node in tag.TraverseIf(predicate))
                    yield return node;

            yield return this;
        }

        private IEnumerable<ProcessTag> TraverseTree()
        {
            var visited = new HashSet<ProcessTag>();
            return TraverseIf((tag) =>
            {
                if (visited.Contains(tag))
                    return false;

                visited.Add(tag);
                return true;
            });
        }

        public IList<ProcessTag> Nodes()
        {
            return TraverseTree().ToList();
        }

        public IList<ProcessTag> EndNodes()
        {
            return TraverseTree().Where(t => t.IsEndNode()).ToList();
        }

        #endregion

        #region String Operations

        public override string ToString()
        {
            return Label ?? "";
        }

        public static implicit operator ProcessTag(string label)
        {
            return new StringTag(label);
        }

        #endregion

        private class BottomTag : StringTag
        {
            public BottomTag() : base("⊥", -1) { }

            protected override ISet<ProcessTag> InputTags
            {
                get { return new HashSet<ProcessTag> {this}; }
            }
        }
    }

    public class EmptyTag : ProcessTag
    {
        public override string Label { get { return ""; } }

        public sealed override int Verbosity { get { return 1000; } }
    }

    public class HubTag : EmptyTag
    {
        public HubTag(params ProcessTag[] inputs)
        {
            AddInputs(inputs);
        }
    }

    public class JunctionTag : ProcessTag
    {
        private readonly ProcessTag m_Junction;
        private readonly ProcessTag m_Description;

        public override string Label { get { return "JUNCTION"; } }
        public override int Verbosity { get { return -10; } }

        public JunctionTag(ProcessTag description, params ProcessTag[] junctions)
        {
            m_Junction = new HubTag(junctions);
            m_Description = description;
        }

        public override ISet<ProcessTag> SubGraphNodes(Func<ProcessTag, bool> predicate)
        {
            var subNodes = base.SubGraphNodes(predicate);
            if (predicate(m_Description))
                subNodes.UnionWith(m_Junction.SubGraphNodes(predicate));
            return subNodes;
        }
    }

    public class StringTag : ProcessTag
    {
        private readonly string m_Label;
        private readonly int m_Verbosity;

        public override string Label
        {
            get { return m_Label; }
        }

        public override int Verbosity
        {
            get { return m_Verbosity; }
        }

        public StringTag(string label, int verbosity = 0)
        {
            m_Label = label;
            m_Verbosity = string.IsNullOrEmpty(label) ? 1000 : verbosity;
        }
    }

    public static class TagHelper
    {
        public static void AddJunction(this ITagged tagged, ProcessTag description, ITagged input)
        {
            tagged.Tag.Insert(new JunctionTag(description, input.Tag));
            tagged.Tag.Insert(description);
        }

        public static TTagged GetTag<TTagged>(this TTagged tagged, out ProcessTag tag)
            where TTagged : ITagged
        {
            tag = tagged.Tag;
            return tagged;
        }

        public static TTagged Tagged<TTagged>(this TTagged tagged, ProcessTag tag)
            where TTagged : ITagged
        {
            tagged.Tag.Insert(tag);
            return tagged;
        }

        public static TTagged PrefixTagTo<TTagged>(this TTagged tagged, ProcessTag tag)
            where TTagged : ITagged
        {
            tag.AddPrefix(tagged.Tag);
            return tagged;
        }
    }
}