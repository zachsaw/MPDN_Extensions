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
    /// Represents the (temporary) right to view a particular value
    /// Disposing returns that right, allowing resources to be freed
    /// </summary>
    public interface ILease<out TValue> : IDisposable
    {
        TValue Value { get; }
    }

    public static class LeaseHelper
    {
        #region Classes

        private struct Bound<TInput, TOutput> : ILease<TOutput>
        {
            private ILease<TInput> m_Input;
            private ILease<TOutput> m_Output;

            public Bound(ILease<TInput> input, Func<TInput, ILease<TOutput>> f)
                : this(input, f(input.Value))
            { }

            private Bound(ILease<TInput> input, ILease<TOutput> output)
            {
                m_Input = input;
                m_Output = output;
            }

            #region Lease Implementation

            public TOutput Value
            {
                get { return m_Output.Value; }
            }

            public void Dispose()
            {
                m_Output.Dispose();
                m_Input.Dispose();
            }

            #endregion
        }

        private struct Folded<TValue> : ILease<IEnumerable<TValue>>
        {
            public Folded(params ILease<TValue>[] leases)
            {
                m_Leases = leases;
            }

            #region Lease Implementation

            private readonly ILease<TValue>[] m_Leases;

            public IEnumerable<TValue> Value { get { return m_Leases.Select(x => x.Value).ToList(); } }

            public void Dispose()
            {
                foreach (var lease in m_Leases)
                    lease.Dispose();
            }

            #endregion
        }

        public struct Just<TValue> : ILease<TValue>
        {
            private readonly TValue m_Value;

            public Just(TValue value)
            {
                m_Value = value;
            }

            public TValue Value { get { return m_Value; } }

            public void Dispose() { }
        }

        #endregion

        #region Extension Methods

        public static void Extract<A>(this ILease<A> lease, Action<A> callback)
        {
            using (lease)
                callback(lease.Value);
        }

        public static void ExtractAll<A>(this IEnumerable<ILease<A>> leases, Action<IEnumerable<A>> action)
        {
            leases.Fold().Extract(action);
        }

        public static ILease<IEnumerable<A>> Fold<A>(this IEnumerable<ILease<A>> leases)
        {
            return new Folded<A>(leases.ToArray());
        }

        #endregion

        #region Monad Implementation

        public static ILease<A> Return<A>(A value)
        {
            return new Just<A>(value);
        }

        public static ILease<B> Bind<A, B>(this ILease<A> lease, Func<A, ILease<B>> f)
        {
            return new Bound<A, B>(lease, f);
        }

        public static ILease<B> Map<A, B>(this ILease<A> lease, Func<A, B> f)
        {
            return lease.Bind(x => Return(f(x)));
        }

        #endregion
    }
}