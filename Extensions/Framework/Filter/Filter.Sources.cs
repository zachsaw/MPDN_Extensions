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

using System.Collections.Generic;

namespace Mpdn.Extensions.Framework.Filter
{
    public abstract class BaseSourceFilter<TOutput> : Filter<IFilterOutput, TOutput>
        where TOutput : class, IFilterOutput
    {
        protected BaseSourceFilter()
        {
            Tag = FilterTag.Bottom;
        }

        protected override void Render(IList<IFilterOutput> inputs) { }
    }

    public class SourceFilter<TOutput> : BaseSourceFilter<TOutput>
        where TOutput : class, IFilterOutput
    {
        private readonly TOutput m_Output;

        public SourceFilter(TOutput output)
        {
            m_Output = output;
        }

        protected override TOutput DefineOutput()
        {
            return m_Output;
        }
    }
}