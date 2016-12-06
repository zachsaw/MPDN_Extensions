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
using Mpdn.Extensions.Framework.Chain;

namespace Mpdn.Extensions.Framework.Filter
{
    public interface IPinFilter<in TFilter> 
        where TFilter : IFilter<IFilterOutput>
    {
        void ConnectPinTo(TFilter filter);
    }

    public abstract class PinFilter<TOutput> : Filter<TOutput, TOutput>, IPinFilter<IFilter<TOutput>> 
        where TOutput : class, IFilterOutput
    {
        protected IFilter<TOutput> Pin { get { return InputFilters[0]; } }

        protected PinFilter(TOutput output) : base(output, new FilterPin()) { }

        public void ConnectPinTo(IFilter<TOutput> filter)
        {
            var filterPin = Pin as FilterPin;
            if (filterPin == null)
                throw new InvalidOperationException("Can't reconnect pin when already compiled.");

            filterPin.ConnectTo(filter);
        }

        protected class FilterPin : Filter<IFilterOutput, TOutput>
        {
            private IFilter<TOutput> m_PinInput;

            // WARNING: Output undefined until compiled
            public FilterPin() : base(null) { }

            public void ConnectTo(IFilter<TOutput> input)
            {
                if (m_PinInput != null)
                    throw new InvalidOperationException("Pin can only be connected once.");

                m_PinInput = input;
            }

            protected override void Render(IList<IFilterOutput> inputs)
            {
                throw new InvalidOperationException("Uncompiled filter.");
            }

            protected override IFilter<TOutput> Optimize()
            {
                if (m_PinInput == null)
                    throw new InvalidOperationException("Pin Hasn't been connected yet.");

                Tag.AddPrefix(m_PinInput.Tag);
                return m_PinInput.Compile();
            }
        }
    }

    public abstract class PinFilterChain<TFilter, TClass> : Chain<TClass>
        where TFilter : IPinFilter<TClass>, TClass
        where TClass : IFilter<IFilterOutput>
    {
        public override TClass Process(TClass input)
        {
            return MakeFilter(input);
        }

        protected abstract TFilter MakeFilter();

        protected virtual TClass MakeFilter(TClass input)
        {
            var filter = MakeFilter();
            filter.ConnectPinTo(input);
            return filter;
        }
    }

    public class StaticPinFilterChain<TFilter, TClass> : PinFilterChain<TFilter, TClass>
        where TFilter : IPinFilter<TClass>, TClass, new()
        where TClass : IFilter<IFilterOutput>
    {
        protected override TFilter MakeFilter()
        {
            return new TFilter();
        }
    }
}