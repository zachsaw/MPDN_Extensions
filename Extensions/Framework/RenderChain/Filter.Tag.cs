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
using IBaseFilter = Mpdn.Extensions.Framework.RenderChain.IFilter<Mpdn.IBaseTexture>;

namespace Mpdn.Extensions.Framework.RenderChain
{
    public abstract class FilterTag
    {
        public static readonly FilterTag Bottom = new BottomTag();

        public abstract string Label { get; }

        #region Base Implementation

        private readonly List<IBaseFilter> m_FilterInputs = new List<IBaseFilter>();
        private List<FilterTag> m_InputTags = new List<FilterTag>();

        private int m_Index;
        protected IEnumerable<FilterTag> SubTags { get; private set; }

        protected bool Initialized { get { return m_Index != 0; } }

        public virtual bool IsEmpty()
        {
            return string.IsNullOrEmpty(Label);
        }

        public virtual int Initialize(int count = 1)
        {
            if (Initialized)
                return count;

            var tags = new List<FilterTag>();
            m_Index = count;
            m_InputTags = m_InputTags.Concat(
                    m_FilterInputs
                    .Select(t => t.Tag))
                    .Distinct()
                    .ToList();

            foreach (var input in m_InputTags)
            {
                m_Index = input.Initialize(m_Index);

                if (input.IsEmpty())
                    tags.AddRange(input.SubTags);
                else
                    tags.Add(input);
            }
            
            SubTags = tags.Distinct().ToList();
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

        public void AddInput(IBaseFilter filter)
        {
            m_FilterInputs.Add(filter);
        }

        public void AddInputLabel(FilterTag tag)
        {
            m_InputTags.Add(tag);
        }

        public bool HasAncestor(FilterTag ancestor)
        {
            return (this == ancestor)
                   || (     m_Index > ancestor.m_Index
                       &&   m_InputTags.Any(t => t.HasAncestor(ancestor)));
        }

        public bool ConnectedTo(FilterTag tag)
        {
            return (m_Index < tag.m_Index)
                ? tag.HasAncestor(this)
                : this.HasAncestor(tag);
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
            public BottomTag() : base("_|_") { }

            public override int Initialize(int count = 1)
            {
                m_FilterInputs.Clear();
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
        public override string Label { get; }

        public StringTag(string label)
        {
            Label = label;
        }
    }

    public class HiddenTag : StringTag
    {
        private bool m_Printing;

        public HiddenTag(string label) : base(label) { }

        public override string Label
        {
            get { return m_Printing ? "" : base.Label; }
        }

        public override string CreateString(int minIndex = -1)
        {
            Initialize();

            m_Printing = true;
            var result = base.CreateString(minIndex);
            m_Printing = false;
            return result;
        }
    }

    public class TemporaryTag : StringTag
    {
        public TemporaryTag(string label) : base(label) { }

        public override bool IsEmpty() { return Initialized; }
    }

    public class ChromaScalerTag : StringTag
    {
        private readonly IBaseFilter m_ChromaFilter;

        private FilterTag m_ChromaTag;

        public ChromaScalerTag(IBaseFilter chromaFilter, string label)
            : base(label)
        {
            m_ChromaFilter = chromaFilter;

            AddInput(m_ChromaFilter);
        }

        public override int Initialize(int count = 1)
        {
            if (!Initialized)
            {
                m_ChromaTag = m_ChromaFilter.Tag;
            }

            return base.Initialize(count);
        }

        public override string CreateString(int minIndex = -1)
        {
            Initialize();

            var lumaPart = new EmptyTag();
            var chromaPart = new StringTag(Label);

            foreach (var tag in SubTags)
                if (tag.ConnectedTo(m_ChromaTag))
                    chromaPart.AddInputLabel(tag);
                else
                    lumaPart.AddInputLabel(tag);

            lumaPart.Initialize();
            chromaPart.Initialize();

            var luma = lumaPart
                .CreateString(minIndex)
                .FlattenStatus()
                .PrependToStatus("Luma: ");

            var chroma = chromaPart
                .CreateString(minIndex)
                .FlattenStatus();
            if (!chroma.StartsWith("Chroma: "))
                chroma = chroma.PrependToStatus(luma == "" ? "" : "Chroma: ");

            return chroma.AppendStatus(luma);
        }
    }

    public static class TagHelper
    {
        public static bool IsNullOrEmpty(this FilterTag tag)
        {
            return tag == null || tag.IsEmpty();
        }

        public static FilterTag Append(this FilterTag tag, FilterTag newTag)
        {
            if (tag != null && newTag != null)
                newTag.AddInputLabel(tag);

            return newTag ?? tag;
        }
    }
}