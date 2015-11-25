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

    public interface IFilter<out TOutput> : IDisposable
        where TOutput : class, IFilterOutput
    {
        TOutput Output { get; }

        int LastDependentIndex { get; }
        void Render();
        void Reset();
        void Initialize(int time = 1);
        IFilter<TOutput> Compile();

        void AddTag(FilterTag newTag);
        FilterTag Tag { get; }
    }

    public interface IFilterOutput : IDisposable
    {
        void Allocate();
        void Deallocate();
    }

    public abstract class FilterOutput : IFilterOutput
    {
        public abstract void Allocate();

        public abstract void Deallocate();

        #region Resource Management

        ~FilterOutput()
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
            Deallocate();
        }

        #endregion
    }

    public abstract class Filter<TInput, TOutput> : IFilter<TOutput>
        where TOutput : class, IFilterOutput
        where TInput : class, IFilterOutput
    {
        protected Filter(params IFilter<TInput>[] inputFilters)
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

        protected abstract void Render(IList<TInput> inputs);

        protected abstract TOutput DefineOutput();

        #region IFilter Implementation

        private IFilter<TInput>[] m_OriginalInputFilters;

        private bool m_Updated;
        private bool m_Initialized;
        private int m_FilterIndex;

        private IFilter<TOutput> m_CompilationResult;
        private TOutput m_Output;

        public IFilter<TInput>[] InputFilters { get; private set; }

        public TOutput Output
        {
            get { return m_Output ?? DefineOutput(); }
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

        public IFilter<TOutput> Compile()
        {
            if (m_CompilationResult != null)
                return m_CompilationResult;

            m_Output = Output;

            m_OriginalInputFilters = InputFilters;
            InputFilters = InputFilters
                .Select(x => x.Compile())
                .ToArray();

            m_CompilationResult = Optimize();
            return m_CompilationResult;
        }

        public FilterTag Tag { get; protected set; }

        public void AddTag(FilterTag newTag)
        {
            Tag = Tag.Append(newTag);
        }

        protected virtual IFilter<TOutput> Optimize()
        {
            return this;
        }

        public virtual void Render()
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
                    .Select(f => f.Output)
                    .ToList();

            Output.Allocate();

            Render(inputTextures);

            foreach (var filter in InputFilters)
            {
                if (filter.LastDependentIndex <= m_FilterIndex)
                {
                    filter.Reset();
                }
            }
        }

        public virtual void Reset()
        {
            m_Updated = false;

            Output.Deallocate();
        }

        #endregion

        #region Resource Management

        ~Filter()
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

            DisposeHelper.DisposeElements(ref m_OriginalInputFilters);
            DisposeHelper.DisposeElements(InputFilters);
            DisposeHelper.Dispose(Output);
            InputFilters = null;
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