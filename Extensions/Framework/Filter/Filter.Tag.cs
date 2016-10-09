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

namespace Mpdn.Extensions.Framework.Filter
{
    using IBaseFilter = IFilter<IFilterOutput>;

    public interface ITaggableFilter<out TOutput> : IFilter<TOutput> 
        where TOutput : class, IFilterOutput
    {
        void EnableTag();
    }

    public abstract class FilterTag
    {
        public static readonly FilterTag Bottom = new BottomTag();

        public abstract string Label { get; }

        public abstract int Verbosity { get; }

        #region Base Implementation

        private readonly HashSet<FilterTag> m_InputTags = new HashSet<FilterTag>();

        protected virtual ISet<FilterTag> InputTags
        {
            get { return m_InputTags; }
        }

        #endregion

        #region Text rendering

        public IList<KeyValuePair<FilterTag, HashSet<FilterTag>>> SubTree(Func<FilterTag, bool> predicate)
        {
            var nodes = Nodes();

            var subNodes = nodes.ToDictionary(n => n, n => new HashSet<FilterTag>());
            foreach (var node in nodes)
            {
                subNodes[node].UnionWith(
                    node.InputTags
                        .SelectMany(x => predicate(x)
                            ? new HashSet<FilterTag> { x }
                            : subNodes[x]));
            }

            return nodes.Where(predicate)
                .Select(n => new KeyValuePair<FilterTag, HashSet<FilterTag>>(n, subNodes[n]))
                .ToList();
        }

#if DEBUG
        protected const int DefaultVerbosity = 0;
        protected string TreeDescription { get { return CreateString(100); } }
#else
        protected const int DefaultVerbosity = 0;
#endif

        public string CreateString(int verbosity = DefaultVerbosity)
        {
            var subtree = SubTree(n => n.Verbosity <= verbosity);
            var subNodes = subtree.ToDictionary(n => n.Key, n => n.Value);
            var nodes = subNodes.Keys;

            var index = subtree
                .Select((n, i) => new KeyValuePair<FilterTag, int>(n.Key, i))
                .ToDictionary(x => x.Key, x => x.Value);

            var minIndex = subNodes.ToDictionary(x => x.Key, x => x.Value
                .Union(new[] { x.Key })
                .Select(n => index[n])
                .Min());

            var depth = nodes.ToDictionary(
                node => node, 
                node => nodes.Count(n => minIndex[n] < index[node] && index[node] < index[n]));

            var labels = nodes
                .Where(n => n.Verbosity >= 0)
                .Select(n => String.Concat(Enumerable.Repeat("    ", depth[n])) + n.Label);
            return string.Join("\n", labels);
        }

        #endregion

        #region Graph Operations

        public void AddInput(FilterTag tag)
        {
            InputTags.Add(tag);
        }

        public void AddInputs(IEnumerable<FilterTag> tags)
        {
            foreach (var tag in tags)
                AddInput(tag);
        }

        public void RemoveInput(FilterTag tag)
        {
            m_InputTags.Remove(tag);
        }

        public void RemoveInputs(IEnumerable<FilterTag> tags)
        {
            foreach (var tag in tags)
                RemoveInput(tag);
        }

        public void AddPrefix(FilterTag prefix)
        {
            foreach (var tag in EndNodes())
                tag.AddInput(prefix);
        }

        public void Insert(FilterTag tag)
        {
            foreach (var input in InputTags)
                tag.InputTags.Add(input);
            InputTags.Clear();
            InputTags.Add(tag);
        }

        public bool IsEndNode()
        {
            return !InputTags.Any();
        }

        public bool HasAncestor(FilterTag ancestor)
        {
            return ancestor.Traverse().Contains(this);
        }

        public bool ConnectedTo(FilterTag tag)
        {
            return tag.HasAncestor(this) || this.HasAncestor(tag);
        }

        #endregion

        #region Node Enumeration

        private IEnumerable<FilterTag> Traverse(ISet<FilterTag> visited)
        {
            visited.Add(this);
            foreach (var tag in InputTags)
            {
                if (visited.Contains(tag))
                    continue;
                foreach (var node in tag.Traverse(visited))
                    yield return node;
            }
            yield return this;
        }

        private IEnumerable<FilterTag> Traverse()
        {
            var visited = new HashSet<FilterTag>();
            return Traverse(visited);
        }

        public IList<FilterTag> Nodes()
        {
            return Traverse().ToList();
        }

        public IList<FilterTag> EndNodes()
        {
            return Traverse().Where(t => t.IsEndNode()).ToList();
        }

        #endregion

        #region String Operations

        public override string ToString()
        {
            return Label ?? "";
        }

        public static implicit operator FilterTag(string label)
        {
            return new StringTag(label);
        }

        #endregion

        private class BottomTag : StringTag
        {
            public BottomTag() : base("⊥", -1) { }

            protected override ISet<FilterTag> InputTags
            {
                get { return new HashSet<FilterTag> {this}; }
            }
        }
    }

    public class EmptyTag : FilterTag
    {
        public override string Label { get { return ""; } }

        public sealed override int Verbosity { get { return 100; } }
    }

    public class StringTag : FilterTag
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
            m_Verbosity = string.IsNullOrEmpty(label) ? 100 : verbosity;
        }
    }

    public static class TagHelper
    {
        public static void AddJunction(this FilterTag from, string description, FilterTag to)
        {
            FilterTag tag = description;
            from.Insert(tag);
            tag.AddInput(to);
        }

        public static TFilter GetTag<TFilter>(this TFilter filter, out FilterTag tag)
            where TFilter : IBaseFilter
        {
            tag = filter.Tag;
            return filter;
        }

        public static TFilter Tagged<TFilter>(this TFilter filter, FilterTag tag)
            where TFilter : IBaseFilter
        {
            filter.Tag.Insert(tag);
            return filter;
        }

        public static TFilter PrefixTagTo<TFilter>(this TFilter filter, FilterTag tag)
            where TFilter : IBaseFilter
        {
            tag.AddPrefix(filter.Tag);
            return filter;
        }

        public static TFilter MakeTagged<TFilter>(this TFilter filter)
            where TFilter : IFilter<IFilterOutput>
        {
            var taggableFilter = filter as ITaggableFilter<IFilterOutput>;
            if (taggableFilter != null)
                taggableFilter.EnableTag();

            return filter;
        }
    }
}