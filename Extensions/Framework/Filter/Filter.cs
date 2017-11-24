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

using Shiandow.Lending;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Mpdn.Extensions.Framework.Filter
{
    using static FilterBaseHelper;

    public interface IFilter<out TOutput, out TValue> : IFilterBase<IFilterOutput<TOutput, TValue>>
    {
        new TOutput Output { get; }
    }

    public class Filter<TOutput, TValue> : FilterBase<IFilterOutput<TOutput, TValue>>, IFilter<TOutput, TValue>
    {
        new public TOutput Output { get { return base.Output.Description; } }

        public Filter(IFilterBase<IFilterOutput<TOutput, TValue>> filterBase)
            : base(filterBase)
        { }
    }

    public static class FilterHelper
    {
        #region Classes

        private class Memoized<TOutput> : ILendable<TOutput>, IDisposable
        {
            private readonly Func<ILease<TOutput>> m_Func;

            public Memoized(Func<ILease<TOutput>> func)
            {
                m_Func = func;
            }

            protected virtual void Initialise(TOutput value) { }

            #region Implementation

            private ILease<TOutput> m_Lease;

            public ILease<TOutput> GetLease()
            {
                if (m_Lease == null)
                {
                    m_Lease = m_Func();
                    Initialise(m_Lease.Value);
                }
                return LeaseHelper.Return(m_Lease.Value);
            }

            public virtual void Dispose()
            {
                DisposeHelper.Dispose(m_Lease);
            }

            #endregion
        }

        private class CheckedFilterOutput<TCheck, TDescription, TValue> : Memoized<IFilterBase<IFilterOutput<TDescription, TValue>>>, IFilterOutput<TDescription, TValue>
            where TCheck : TDescription, IEquatable<TDescription>
        {
            private readonly TCheck m_Description;

            public CheckedFilterOutput(TCheck description, ILendable<IFilterBase<IFilterOutput<TDescription, TValue>>> compiler) 
                : base(compiler.GetLease)
            {
                m_Description = description;
            }

            protected override void Initialise(IFilterBase<IFilterOutput<TDescription, TValue>> value)
            {
                base.Initialise(value);
                if (!m_Description.Equals(value.Output.Description))
                    throw new InvalidOperationException("Generated output does not match description.");
            }

            #region IFilterOutput Implementation

            public TDescription Description { get { return m_Description; } }

            ILease<TValue> ILendable<TValue>.GetLease()
            {
                return GetLease().Bind(x => x.Output.GetLease());
            }

            #endregion
        }

        private class CompiledFilter<TCheck, TDescription, TValue> : CheckedFilterOutput<TCheck, TDescription, TValue>, IFilter<TDescription, TValue>
            where TCheck : TDescription, IEquatable<TDescription>
        {
            public CompiledFilter(TCheck description, ILendable<IFilterBase<IFilterOutput<TDescription, TValue>>> compiler)
                : base(description, compiler)
            { }

            #region Implementation

            private IFilterBase<IFilterOutput<TDescription, TValue>> m_FilterBase;

            protected override void Initialise(IFilterBase<IFilterOutput<TDescription, TValue>> value)
            {
                base.Initialise(value);

                m_ProcessData.AddInput(value.ProcessData);
                m_FilterBase = value;
            }

            #endregion

            #region IFilterBase Implementation

            private readonly IProcessData m_ProcessData = new ProcessData();

            public IProcessData ProcessData { get { return m_ProcessData; } }
            public TDescription Output { get { return Description; } }

            IFilterOutput<TDescription, TValue> IFilterBase<IFilterOutput<TDescription, TValue>>.Output { get { return this; } }

            public override void Dispose()
            {
                base.Dispose();
                DisposeHelper.Dispose(ref m_FilterBase);
            }

            #endregion
        }

        #endregion

        public static IFilter<A,X> Return<A,X>(IFilterBase<IFilterOutput<A, X>> filterBase)
        {
            return new Filter<A,X>(filterBase);
        }

        public static IFilter<B, Y> Compile<A, B, Y>(A description, ILendable<IFilterBase<IFilterOutput<B, Y>>> compiler)
            where A : B, IEquatable<B>
        {
            return new CompiledFilter<A, B, Y>(description, compiler);
        }

        public static IFilter<B, Y> Compile<A, B, Y>(A description, Func<IFilterBase<IFilterOutput<B, Y>>> compiler)
            where A : B, IEquatable<B>
        {
            return Compile(description, compiler.MakeDeferred());
        }

        public static void Extract<X>(this IFilterBase<ILendable<X>> filter, Action<X> callback)
        {
            filter.Output.Extract(callback);
        }

        public static Y Extract<X,Y>(this IFilterBase<ILendable<X>> filter, Func<X,Y> callback)
        {
            return filter.Output.Extract(callback);
        }

        public static IFilter<IEnumerable<A>, IEnumerable<X>> Fold<A, X>(this IEnumerable<IFilter<A, X>> filters)
        {
            return Return(from values in FilterBaseHelper.Fold(filters)
                          let outputs = filters.Select(x => x.Output)
                          select FilterOutputHelper.Return(outputs, values.Fold()));
        }
    }
}