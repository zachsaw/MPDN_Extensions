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
// License along with this library

using System;
using Mpdn.Extensions.Framework.Filter;

namespace Mpdn.Extensions.Framework.Chain
{
    public abstract class FilterChain<TFilter> : Chain<TFilter>
        where TFilter : class, IFilter<IFilterOutput>
    {
        protected abstract TFilter CreateFilter(TFilter input);

        public virtual string Description
        {
            get { return GetType().Name; }
        }

        public override TFilter Process(TFilter input)
        {
            TFilter output = CreateFilter(input);
            
            if (output == input)
                return input;

            output.Tag.Insert(Description);

            return output;
        }
    }
}
