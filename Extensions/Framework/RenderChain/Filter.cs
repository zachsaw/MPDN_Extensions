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
using Mpdn.RenderScript;
using IBaseFilter = Mpdn.Extensions.Framework.RenderChain.IFilter<Mpdn.IBaseTexture>;

namespace Mpdn.Extensions.Framework.RenderChain
{
    public interface IFilter<out TTexture>
        where TTexture : class, IBaseTexture
    {
        TTexture OutputTexture { get; }
        TextureSize OutputSize { get; }
        TextureFormat OutputFormat { get; }
        int LastDependentIndex { get; }
        void Render();
        void Reset();
        void Initialize(int time = 1);
        IFilter<TTexture> Compile();

        void AddTag(FilterTag newTag);
        FilterTag Tag { get; }
    }

    public interface IFilter : IFilter<ITexture2D>
    {
    }

    public interface IResizeableFilter : IFilter
    {
        void SetSize(TextureSize outputSize);
        void MakeTagged();
    }

    public abstract class Filter : IFilter
    {
        protected Filter(params IBaseFilter[] inputFilters)
        {
            if (inputFilters == null || inputFilters.Any(f => f == null))
            {
                throw new ArgumentNullException("inputFilters");
            }

            m_Initialized = false;
            m_CompilationResult = null;
            InputFilters = inputFilters;

            Tag = new EmptyTag();

            foreach (var filter in inputFilters)
                Tag.AddInput(filter);
        }

        protected abstract void Render(IList<IBaseTexture> inputs);

        public abstract TextureSize OutputSize { get; }

        #region IFilter Implementation

        private bool m_Updated;
        private bool m_Initialized;
        private int m_FilterIndex;
        private IFilter<ITexture2D> m_CompilationResult;

        protected ITargetTexture OutputTarget { get; private set; }

        public IBaseFilter[] InputFilters { get; }

        public ITexture2D OutputTexture { get { return OutputTarget; } }

        public virtual TextureFormat OutputFormat
        {
            get { return Renderer.RenderQuality.GetTextureFormat(); }
        }
        
        public int LastDependentIndex { get; private set; }

        public void Initialize(int time = 1)
        {
            LastDependentIndex = time;

            if (m_Initialized)
                return;

            foreach (var f in InputFilters)
            {
                f.Initialize(LastDependentIndex);
                LastDependentIndex = f.LastDependentIndex;
            }

            m_FilterIndex = LastDependentIndex;

            foreach (var filter in InputFilters)
            {
                filter.Initialize(m_FilterIndex);
            }

            LastDependentIndex++;
            m_Initialized = true;
        }

        public IFilter<ITexture2D> Compile()
        {
            if (m_CompilationResult != null)
                return m_CompilationResult;

            for (int i = 0; i < InputFilters.Length; i++)
            {
                InputFilters[i] = InputFilters[i].Compile();
            }

            m_CompilationResult = Optimize();
            return m_CompilationResult;
        }

        public FilterTag Tag { get; protected set; }

        public void AddTag(FilterTag newTag)
        {
            Tag = Tag.Append(newTag);
        }

        protected virtual IFilter<ITexture2D> Optimize()
        {
            return this;
        }

        public void Render()
        {
            if (m_Updated)
                return;

            m_Updated = true;

            foreach (var filter in InputFilters)
            {
                filter.Render();
            }

            var inputTextures =
                InputFilters
                    .Select(f => f.OutputTexture)
                    .ToList();

            OutputTarget = TexturePool.GetTexture(OutputSize, OutputFormat);

            Render(inputTextures);

            foreach (var filter in InputFilters)
            {
                if (filter.LastDependentIndex <= m_FilterIndex)
                {
                    filter.Reset();
                }
            }
        }

        public void Reset()
        {
            m_Updated = false;

            if (OutputTarget != null)
            {
                TexturePool.PutTexture(OutputTarget);
            }

            OutputTarget = null;
        }

        #endregion
    }

    public static class FilterHelper
    {
        public static TFilter InitializeFilter<TFilter>(this TFilter filter)
            where TFilter: IBaseFilter
        {
            filter.Initialize();
            return filter;
        }

        public static TFilter GetTag<TFilter>(this TFilter filter, out FilterTag tag)
            where TFilter : IBaseFilter
        {
            tag = new EmptyTag();
            tag.AddInput(filter);
            return filter;
        }

        public static TFilter Tagged<TFilter>(this TFilter filter, FilterTag tag)
            where TFilter : IBaseFilter
        {
            filter.AddTag(tag);
            return filter;
        }
    }
}