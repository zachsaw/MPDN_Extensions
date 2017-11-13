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
    public interface IProcess<in TInput, out TOutput>
    {
        TOutput ApplyTo(TInput input);
    }

    public interface IMultiProcess<in TInput, out TOutput> : IProcess<IEnumerable<TInput>, TOutput> { }

    public static class ProcessHelper
    {
        public static TOutput ApplyTo<TInput, TOutput>(this IMultiProcess<TInput, TOutput> process, params TInput[] inputs)
        {
            return process.ApplyTo(inputs);
        }

        public static TOutput Apply<TInput, TOutput>(this TInput input, IProcess<TInput, TOutput> process)
        {
            return process.ApplyTo(input);
        }
    }
}