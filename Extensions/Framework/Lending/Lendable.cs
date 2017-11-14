// Copyright Shiandow (2017), some rights reserved

// This file is free software; you can redistribute it and/or
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

namespace Shiandow.Lending
{
    /// <summary>
    /// Represents a value that can be 'leased'
    /// </summary>
    public interface ILendable<out TValue>
    {
        ILease<TValue> GetLease();
    }

    /// <summary>
    /// Simple reference counting implementation
    /// Exposes methods for allocating, calculating, deallocating the value as needed
    /// </summary>
    public abstract class Lendable<TValue> : ILendable<TValue>
    {
        protected abstract void Allocate();

        protected abstract TValue Value { get; }

        protected abstract void Deallocate();

        #region ILeaseable Implementation

        // Counted reference
        private struct Lease : ILease<TValue>
        {
            private readonly Lendable<TValue> m_Owner;

            public Lease(Lendable<TValue> owner)
            {
                m_Owner = owner;
                m_Disposed = false;
            }

            public TValue Value { get { return m_Owner.Value; } }

            #region Resource Management

            private bool m_Disposed; // To detect redundant calls

            public void Dispose()
            {
                if (!m_Disposed)
                {
                    m_Disposed = true;
                    m_Owner.Release();
                }
            }

            #endregion
        }

        // Reference counter
        private int m_Leases = 0;

        public ILease<TValue> GetLease()
        {
            if (m_Leases <= 0)
                Allocate();

            m_Leases += 1;
            return new Lease(this);
        }

        private void Release()
        {
            m_Leases -= 1;
            if (m_Leases <= 0)
                Deallocate();
        }

        #endregion
    }

    public static class LendableHelper
    {
        #region Classes

        public class Deferred<TOutput> : Lendable<TOutput>
        {
            private readonly Func<ILease<TOutput>> m_Func;

            public Deferred(Func<ILease<TOutput>> func)
            {
                m_Func = func;
            }

            #region ILendable Implementation

            private ILease<TOutput> m_Lease;

            protected override TOutput Value { get { return m_Lease.Value; } }

            protected override void Allocate()
            {
                m_Lease = m_Func();
            }

            protected override void Deallocate()
            {
                m_Lease.Dispose();
            }

            #endregion
        }

        private class Mapped<TInput, TOutput> : Deferred<TOutput>
        {
            public Mapped(ILendable<TInput> lendable, Func<ILease<TInput>, ILease<TOutput>> func)
                : base(() => func(lendable.GetLease()))
            { }
        }

        private class Bound<TInput, TOutput> : Mapped<TInput, ILendable<TOutput>>, ILendable<TOutput>
        {
            public Bound(ILendable<TInput> lendable, Func<ILease<TInput>, ILease<ILendable<TOutput>>> func) 
                : base(lendable, func)
            { }

            ILease<TOutput> ILendable<TOutput>.GetLease()
            {
                return GetLease().Bind(x => x.GetLease());
            }
        }

        private class Folded<TValue> : Deferred<IReadOnlyList<TValue>>
        {
            public Folded(IReadOnlyList<ILendable<TValue>> lendables)
                : base(() => lendables.Select(x => x.GetLease()).Fold())
            { }
        }

        public class Just<TValue> : ILendable<TValue>
        {
            private readonly TValue m_Value;

            public Just(TValue value)
            {
                m_Value = value;
            }

            public ILease<TValue> GetLease()
            {
                return LeaseHelper.Return(m_Value);
            }
        }

        public class Lazy<TValue> : Lendable<TValue>
        {
            private readonly Func<ILease<TValue>> m_Func;

            public Lazy(ILendable<TValue> lendable)
                : this(() => lendable.GetLease())
            { }

            public Lazy(Func<ILease<TValue>> func)
            {
                m_Func = func;
            }

            #region Lendable Implementation

            private ILease<TValue> m_Lease;

            protected sealed override TValue Value
            {
                get
                {
                    m_Lease = m_Lease ?? m_Func();
                    return m_Lease.Value;
                }
            }

            protected sealed override void Allocate() { }

            protected sealed override void Deallocate()
            {
                using (m_Lease)
                    m_Lease = null;
            }

            #endregion
        }

        #endregion

        #region Extension Methods

        public static void Extract<A>(this ILendable<A> lendable, Action<A> callback)
        {
            lendable.GetLease().Extract(callback);
        }

        public static B Extract<A, B>(this ILendable<A> lendable, Func<A, B> callback)
        {
            return lendable.GetLease().Extract(callback);
        }

        public static void ExtractAll<A>(this IEnumerable<ILendable<A>> lendables, Action<IEnumerable<A>> action)
        {
            lendables
                .Select(x => x.GetLease())
                .ExtractAll(action);
        }

        public static ILendable<IReadOnlyList<A>> Fold<A>(this IEnumerable<ILendable<A>> lendables)
        {
            return new Folded<A>(lendables.ToList());
        }

        #endregion

        #region Monad Implementation

        public static ILendable<A> Return<A>(A value)
        {
            return new Just<A>(value);
        }

        public static ILendable<B> BindLease<A, B>(this ILendable<A> lendable, Func<ILease<A>, ILease<ILendable<B>>> f)
        {
            return new Bound<A, B>(lendable, f);
        }

        public static ILendable<B> BindLease<A, B>(this ILendable<A> lendable, Func<ILease<A>, ILendable<B>> f)
        {
            return new Bound<A, B>(lendable, x => LeaseHelper.Return(f(x)));
        }

        public static ILendable<B> Bind<A, B>(this ILendable<A> lendable, Func<A, ILendable<B>> f)
        {
            return new Bound<A, B>(lendable, x => x.Map(f));
        }

        public static ILendable<B> MapLease<A, B>(this ILendable<A> lendable, Func<ILease<A>, ILease<B>> f)
        {
            return new Mapped<A, B>(lendable, f);
        }

        public static ILendable<B> Map<A, B>(this ILendable<A> lendable, Func<A, B> f)
        {
            return new Mapped<A, B>(lendable, x => x.Map(f));
        }

        #endregion

        #region Linq Helpers

        public static ILendable<B> Select<A, B>(this ILendable<A> lendable, Func<A, B> f)
        {
            return lendable.Map(f);
        }

        public static ILendable<B> SelectMany<A, B>(this ILendable<A> lendable, Func<A, ILendable<B>> f)
        {
            return lendable.Bind(f);
        }

        public static ILendable<C> SelectMany<A, B, C>(this ILendable<A> lendable, Func<A, ILendable<B>> bind, Func<A, B, C> select)
        {
            return lendable
                .Bind((a) => bind(a)
                .Map ((b) => select(a, b)));
        }

        #endregion

        #region Makers

        public static ILendable<A> MakeLazy<A>(this ILendable<A> lendable)
        {
            return new Lazy<A>(lendable);
        }

        public static ILendable<A> MakeLazy<A>(this Func<A> func)
        {
            return new Lazy<A>(() => LeaseHelper.Return(func()));
        }

        public static ILendable<A> MakeDeferred<A>(this Func<A> func)
        {
            return new Deferred<A>(() => LeaseHelper.Return(func()));
        }

        #endregion
    }
}