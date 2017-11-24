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
        public static IFilter<A,X> Return<A,X>(IFilterBase<IFilterOutput<A, X>> filterBase)
        {
            return new Filter<A,X>(filterBase);
        }

        public static IFilter<IEnumerable<A>, IEnumerable<X>> Fold<A, X>(this IEnumerable<IFilter<A, X>> filters)
        {
            return Return(from values in FilterBaseHelper.Fold(filters)
                          let outputs = filters.Select(x => x.Output)
                          select FilterOutputHelper.Return(outputs, values.Fold()));
        }

        public static void Extract<X>(this IFilterBase<ILendable<X>> filter, Action<X> callback)
        {
            filter.Output.Extract(callback);
        }

        public static Y Extract<X,Y>(this IFilterBase<ILendable<X>> filter, Func<X,Y> callback)
        {
            return filter.Output.Extract(callback);
        }
    }
}