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
// 

using System;

namespace Mpdn.Extensions.Framework.Chain
{
    public abstract class Chain<T>
    {
        public abstract T Process(T input);

        #region Operators

        public static readonly Chain<T> IdentityChain = new StaticChain<T>(x => x);
        
        public static explicit operator Chain<T>(Func<T, T> map)
        {
            return new StaticChain<T>(map);
        }

        public static T operator +(T input, Chain<T> chain)
        {
            return chain.Process(input);
        }

        public static Chain<T> operator +(Chain<T> first, Chain<T> second)
        {
            return (Chain<T>)(Func<T, T>)(x => x + first + second);
        }

        #endregion
    }

    public class StaticChain<T> : Chain<T>
    {
        private readonly Func<T,T> m_Func;

        public StaticChain(Func<T,T> func)
        {
            m_Func = func;
        }

        public override T Process(T input)
        {
            return m_Func(input);
        }
    }
}