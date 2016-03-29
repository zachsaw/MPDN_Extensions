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

        #region Base Implementation

        private int m_Index;
        private readonly HashSet<FilterTag> m_InputTags = new HashSet<FilterTag>();

        protected virtual ISet<FilterTag> InputTags
        {
            get { return m_InputTags; }
        }

        protected IEnumerable<FilterTag> SubTags
        {
            get
            {
                return InputTags
                    .Where(x => x.IsEmpty())
                    .SelectMany(x => x.SubTags)
                    .Concat(InputTags.Where(x => !x.IsEmpty()))
                    .Distinct()
                    .ToArray();
            }
        }

        protected bool Initialized { get { return m_Index != 0; } }

        public virtual bool IsEmpty()
        {
            return string.IsNullOrEmpty(Label);
        }

        public virtual int Initialize(int count = 1)
        {
            if (Initialized)
                return count;
            m_Index = count;

            foreach (var input in InputTags)
                m_Index = input.Initialize(m_Index);

            return ++m_Index;
        }

        #endregion

        #region Text rendering

        public virtual string CreateString(int minIndex = -1)
        {
            Initialize();

            string result = Label;
            var tags = SubTags
                .OrderBy(l => l.m_Index)
                .SkipWhile(l => l.m_Index <= minIndex)
                .ToList();

            if (tags.Any())
            {
                var first = tags.First();
                result = first
                    .CreateString(minIndex)
                    .AppendStatus(Label);
                minIndex = first.m_Index;

                foreach (var tag in tags.Skip(1))
                {
                    result = result.AppendSubStatus(tag.CreateString(minIndex));
                    minIndex = tag.m_Index;
                }
            }
            return result;
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
                InputTags.Add(tag);
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
            return (m_Index < tag.m_Index)
                ? tag.HasAncestor(this)
                : HasAncestor(tag);
        }

        #endregion

        #region Node Enumeration

        private IEnumerable<FilterTag> Traverse(ISet<FilterTag> visited)
        {
            yield return this;
            var nodes = InputTags
                .Except(visited)
                .SelectMany(inputTag => inputTag.Traverse(visited));
            foreach (var tag in nodes)
                yield return tag;
        }

        private IEnumerable<FilterTag> Traverse()
        {
            var visited = new HashSet<FilterTag>();
            foreach (var node in Traverse(visited))
            {
                visited.Add(node);
                yield return node;
            }
        }

        public IEnumerable<FilterTag> Nodes()
        {
            return Traverse().ToList();
        }

        public IEnumerable<FilterTag> EndNodes()
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
            public BottomTag() : base("⊥") { }

            protected override ISet<FilterTag> InputTags
            {
                get { return new HashSet<FilterTag> { this }; }
            }

            public override int Initialize(int count = 1)
            {
                return count;
            }

            public override string CreateString(int minIndex = -1)
            {
                return "";
            }
        }
    }

    public class EmptyTag : FilterTag
    {
        public sealed override string Label { get { return ""; } }
    }

    public class StringTag : FilterTag
    {
        private readonly string m_Label;

        public override string Label
        {
            get { return m_Label; }
        }

        public StringTag(string label)
        {
            m_Label = label;
        }
    }

    public class JunctionTag : EmptyTag
    {
        private readonly FilterTag m_Junction;

        public JunctionTag(FilterTag junction)
        {
            m_Junction = junction;
        }

        public override int Initialize(int count = 1)
        {
            AddInput(m_Junction);
            return base.Initialize(count);
        }
    }

    public static class TagHelper
    {
        public static void AddJunction(this FilterTag tag, FilterTag name, FilterTag to)
        {
            tag.Insert(new JunctionTag(to));
            tag.Insert(name);
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