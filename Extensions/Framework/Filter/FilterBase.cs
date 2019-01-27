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
    public interface IFilterBase : IDisposable, ITagged { }

    public interface IFilterBase<out TOutput> : IFilterBase
    {
        TOutput Output { get; }
    }

    public class FilterBase<TOutput> : IFilterBase<TOutput>
    {
        public FilterBase(IFilterBase<TOutput> filterBase)
        {
            if (filterBase == null)
                throw new ArgumentNullException("FilterBase is not allowed to be null.");

            m_FilterBase = filterBase;
        }

        #region IFilterBase Passthrough

        private readonly IFilterBase<TOutput> m_FilterBase;

        public IProcessData ProcessData { get { return m_FilterBase.ProcessData; } }

        public TOutput Output { get { return m_FilterBase.Output; } }

        #region Resource Management
        
        ~FilterBase() { Dispose(false); }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
                m_FilterBase.Dispose();
        }

        #endregion

        #endregion
    }

    public static class FilterBaseHelper
    {
        #region Classes

        public class Just<TOutput> : IFilterBase<TOutput>
        {
            public Just(TOutput output)
            {
                m_ProcessData = new ProcessData();
                m_Output = output;
            }

            #region IFilterBase Implementation

            private readonly IProcessData m_ProcessData;
            private readonly TOutput m_Output;

            public IProcessData ProcessData { get { return m_ProcessData; } }

            public TOutput Output { get { return m_Output; } }

            public virtual void Dispose() { }

            #endregion
        }

        public class JustBound<TOutput> : Just<TOutput>
        {
            public JustBound(TOutput output) 
                : base(output)
            { }

            #region Resource Management

            private bool m_Disposed = false;

            ~JustBound() { Dispose(false); }

            public override void Dispose()
            {
                Dispose(true);
                GC.SuppressFinalize(this);
            }

            protected void Dispose(bool disposing)
            {
                if (m_Disposed)
                    return;

                m_Disposed = true;
                DisposeHelper.Dispose(Output);
            }

            #endregion
        }

        private struct Bound<TOutput> : IFilterBase<TOutput>
        {
            public Bound(IFilterBase<TOutput> output, params IFilterBase[] inputs)
            {
                m_Disposed = false;

                m_Inputs = inputs;
                m_Output = output;

                foreach (var filter in inputs)
                    ProcessData.AddInput(filter.ProcessData);
            }

            #region IFilterBase Implementation

            private readonly IFilterBase<TOutput> m_Output;
            private readonly IFilterBase[] m_Inputs;

            public IProcessData ProcessData { get { return m_Output.ProcessData; } }

            TOutput IFilterBase<TOutput>.Output { get { return m_Output.Output; } }

            #region Resource Management

            private bool m_Disposed;

            public void Dispose()
            {
                if (m_Disposed)
                    return;

                m_Disposed = true;
                foreach (var input in m_Inputs)
                    input.Dispose();
                m_Output.Dispose();
            }

            #endregion

            #endregion
        }
        
        #endregion

        #region Monad Implementation

        public static IFilterBase<A> Return<A>(A value)
        {
            return new Just<A>(value);
        }

        public static IFilterBase<A> Bind<A>(this A value)
            where A : IDisposable
        {
            return new JustBound<A>(value);
        }

        public static IFilterBase<B> Bind<A, B>(this IFilterBase<A> filter, Func<A, IFilterBase<B>> f)
        {
            return new Bound<B>(f(filter.Output), new[] { filter });
        }

        public static IFilterBase<B> Map<A, B>(this IFilterBase<A> filter, Func<A, B> f)
        {
            return filter.Bind(x => Return(f(x)));
        }

        #endregion

        #region Linq Helpers

        public static IFilterBase<B> Select<A, B>(this IFilterBase<A> filter, Func<A, B> f)
        {
            return filter.Map(f);
        }

        public static IFilterBase<B> SelectMany<A, B>(this IFilterBase<A> filter, Func<A, IFilterBase<B>> f)
        {
            return filter.Bind(f);
        }

        public static IFilterBase<C> SelectMany<A, B, C>(this IFilterBase<A> filter, Func<A, IFilterBase<B>> bind, Func<A, B, C> select)
        {
            return filter
                .Bind((a) => bind(a)
                 .Map((b) => select(a, b)));
        }

        #endregion

        #region Helper Methods

        public static IFilterBase<IEnumerable<A>> Fold<A>(this IEnumerable<IFilterBase<A>> filters)
        {
            return new Bound<IEnumerable<A>>(Return(filters.Select(x => x.Output).ToList()), filters.ToArray());
        }

        #endregion
    }
}