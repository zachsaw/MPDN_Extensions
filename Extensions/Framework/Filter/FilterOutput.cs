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
    using static LeaseHelper;
    using static LendableHelper;

    public interface IFilterOutput<out TValue> : ILendable<TValue> { }

    public interface IFilterOutput<out TDescription, out TValue> : IFilterOutput<TValue>
    {
        TDescription Description { get; }
    }

    public abstract class FilterOutput<TValue> : Lendable<TValue> { }

    public abstract class FilterOutput<TDescription, TValue> : FilterOutput<TValue>, IFilterOutput<TDescription, TValue>
    {
        public abstract TDescription Description { get; }
    }

    public static class FilterOutputHelper
    {
        #region Classes

        private struct Result<TDescription, TValue> : IFilterOutput<TDescription, TValue>
        {
            public Result(TDescription description, ILendable<TValue> value)
            {
                m_Description = description;
                m_Value = value;
            }

            #region Implementation

            private readonly TDescription m_Description;
            private readonly ILendable<TValue> m_Value;

            public TDescription Description { get { return m_Description; } }

            public ILease<TValue> GetLease()
            {
                return m_Value.GetLease();
            }

            #endregion
        }

        #endregion

        #region Rendering Methods

        private static ILease<Y> Do<X, Y>(this ILendable<Y> output, Action<X, Y> render, ILease<X> lease)
        {
            return lease.Extract(value =>
            {
                var target = output.GetLease();
                render(value, target.Value);
                return target;
            });
        }

        private static ILendable<Y> Do<X, Y>(this ILendable<Y> output, Action<X, Y> render, ILendable<X> input)
        {
            return input.MapLease(lease => new Lazy<Y>(() => output.Do(render, lease)));
        }

        #endregion

        #region Extension Methods

        public static IFilterOutput<A, X> Return<A, X>(A description, ILendable<X> value)
        {
            return new Result<A, X>(description, value);
        }

        public static IFilterOutput<B, Y> Do<B, X, Y>(this IFilterOutput<B, Y> output, Action<X, Y> render, ILendable<X> input)
        {
            return Return(output.Description, output.Do<X, Y>(render, input));
        }

        public static IFilterOutput<IEnumerable<A>, IEnumerable<X>> Fold<A,X>(IEnumerable<IFilterOutput<A,X>> outputs)
        {
            return Return(outputs.Select(x => x.Description), outputs.Fold<X>());
        }

        public static void Extract<A>(this ILendable<A> output, Action<A> callback)
        {
            LendableHelper.Extract(output, callback);
        }

        #endregion
    }
}