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

    public interface IFilter<out TOutput> : IDisposable, ITaggedProcess
        where TOutput : class, IFilterOutput
    {
        TOutput Output { get; }

        int LastDependentIndex { get; }
        void Render();
        void Reset();
        void Initialize(int time = 1);
        IFilter<TOutput> Compile();
    }

    public interface IFilterOutput : IDisposable
    {
        void Initialize();
        void Allocate();
        void Deallocate();
    }

    public abstract class FilterOutput : IFilterOutput
    {
        public virtual void Initialize() { }

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
        protected Filter(TOutput output, params IFilter<TInput>[] inputFilters)
        {
            if (inputFilters == null || inputFilters.Any(f => f == null))
            {
                throw new ArgumentNullException("inputFilters");
            }

            m_Initialized = false;
            m_CompilationResult = null;
            m_InputFilters = inputFilters;

            m_Output = output;

            m_ProcessData = new ProcessData();
            ProcessData.AddInputs(InputFilters.Select(f => f.ProcessData));
        }

        protected abstract void Render(IList<TInput> inputs);

        #region IFilter Implementation

        private readonly IFilter<TInput>[] m_InputFilters;
        private IFilter<TInput>[] m_CompiledFilters;

        private readonly TOutput m_Output;

        private bool m_Updated;
        private bool m_Initialized;
        private int m_FilterIndex;
        private IFilter<TOutput> m_CompilationResult;

        public IFilter<TInput>[] InputFilters { get { return m_CompiledFilters ?? m_InputFilters; } }

        public TOutput Output
        {
            get
            {
                return m_Output;
            }
        }

        public int LastDependentIndex { get; private set; }

        public void Initialize(int time = 1)
        {
            LastDependentIndex = time;

            if (m_Initialized)
                return;

            if (m_CompilationResult != this)
                throw new InvalidOperationException("Uncompiled Filter.");

            foreach (var f in InputFilters)
            {
                f.Initialize(LastDependentIndex);
                LastDependentIndex = f.LastDependentIndex;
            }

            Initialize();

            m_FilterIndex = LastDependentIndex;

            foreach (var filter in InputFilters)
            {
                filter.Initialize(m_FilterIndex);
            }

            ProcessData.Rank = m_FilterIndex;
            LastDependentIndex++;
            m_Initialized = true;
        }

        // Called if the filter is actually used, but before it is used.
        protected virtual void Initialize()
        {
            Output.Initialize();
        }

        public IFilter<TOutput> Compile()
        {
            if (m_CompilationResult != null)
                return m_CompilationResult;

            m_CompiledFilters = m_InputFilters
                .Select(x => x.Compile())
                .ToArray();

            m_CompilationResult = Optimize();
            ProcessData.AddInputs(new[] { m_CompilationResult.ProcessData });
            return m_CompilationResult;
        }

        protected virtual IFilter<TOutput> Optimize()
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

            var inputs =
                InputFilters
                    .Select(f => f.Output)
                    .ToList();

            Output.Allocate();

            Render(inputs);

            foreach (var filter in InputFilters.Where(filter => filter.LastDependentIndex <= m_FilterIndex))
            {
                filter.Reset();
            }
        }

        public virtual void Reset()
        {
            m_Updated = false;

            Output.Deallocate();
        }

        #endregion

        #region ITaggedProcess Implementation

        private readonly IProcessData m_ProcessData;

        public IProcessData ProcessData { get { return m_ProcessData; } }

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

            DisposeHelper.DisposeElements(m_InputFilters);
            DisposeHelper.DisposeElements(ref m_CompiledFilters);
            DisposeHelper.Dispose(m_Output);
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

        public static TOther Apply<TFilter, TOther>(this TFilter filter, Func<TFilter, TOther> map)
            where TFilter : IBaseFilter
            where TOther : IBaseFilter
        {
            return map(filter);
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