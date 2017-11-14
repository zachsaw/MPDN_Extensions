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
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;

namespace Shiandow.Merging
{
    public interface IMergeable<in T>
    {
        void MergeWith(T data);
    }

    public static class MergeableHelper
    {
        public static void Add<T>(this IMergeable<IEnumerable<T>> mergeable, T data)
        {
            mergeable.MergeWith(new[] { data });
        }
    }

    public class MergeableCollection<TValue> : Collection<TValue>, IMergeable<IEnumerable<TValue>>
    {
        public void MergeWith(IEnumerable<TValue> data)
        {
            foreach (var x in data) Add(x);
        }
    }
}
