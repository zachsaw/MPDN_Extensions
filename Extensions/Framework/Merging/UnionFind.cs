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
using System.Text;

namespace Shiandow.Merging
{
    public interface IUnionFind<TData> : IMergeable<IUnionFind<TData>>, IMergeable<TData>
    {
        IUnionFind<TData> Root { get; }
        int Depth { get; }

        TData Data { get; }
    }

    public class UnionFind<IData, TData> : IUnionFind<IData>
        where TData : class, IData, IMergeable<IData>, new()
    {
        // Union-find structure augmented with data

        public UnionFind() { m_Root = this; }

        #region IUnionFind Implementation

        private IUnionFind<IData> m_Root;
        private TData m_Data;

        protected TData RootData { get { return m_Data ?? (m_Data = new TData()); } }

        public IData Data { get { return (m_Root == this) ? RootData : Root.Data; } }

        public int Depth { get; private set; }

        public IUnionFind<IData> Root
        {
            get
            {
                if (m_Root != this)
                    m_Root = m_Root.Root;
                return m_Root;
            }

            private set
            {
                if (value != m_Root && m_Data != null)
                {
                    value.MergeWith(m_Data);
                    m_Data = null;
                }
                m_Root = value;
            }
        }

        #endregion

        #region IMergeable Implementation

        public void MergeWith(IData data)
        {
            if (Root != this)
                Root.MergeWith(data);
            else
                RootData.MergeWith(data);
        }

        public void MergeWith(IUnionFind<IData> tree)
        {
            if (Root != this)
                Root.MergeWith(tree.Root);
            else 
                MergeWithRoot(tree.Root);
        }

        private void MergeWithRoot(IUnionFind<IData> root)
        {
            if (Depth < root.Depth)
                Root = root;
            else if (Depth > root.Depth)
                root.MergeWith(this);
            else if (root != this)
            {
                Depth += 1;
                root.MergeWith(this);
            }
        }

        #endregion
    }

    public class UnionCollection<TValue> : UnionFind<IEnumerable<TValue>, MergeableCollection<TValue>> { }
}
