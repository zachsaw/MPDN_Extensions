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
using System.Collections.Generic;
using System.Windows.Forms;
using YAXLib;

namespace Mpdn.Extensions.Framework.Chain
{
    public interface INameable
    {
        string Name { set; }
    }

    public class Preset<T, TScript> : Chain<T>, IChainUi<T, TScript>, INameable
        where TScript : class, IScript
    {
        #region Settings

        private string m_Name;
        private IChainUi<T, TScript> m_Script;

        [YAXAttributeForClass]
        public string Name
        {
            get { return m_Name; }
            set
            {
                m_Name = value;
                if (Script != null && Chain is INameable)
                {
                    ((INameable) Chain).Name = value;
                }
            }
        }

        [YAXAttributeForClass]
        public Guid Guid { get; set; }

        public IChainUi<T, TScript> Script
        {
            get { return m_Script; }
            set
            {
                m_Script = value;
                if (Script != null && Chain is INameable)
                {
                    ((INameable) Chain).Name = Name;
                }
            }
        }

        #endregion

        #region Chain implementation

        public Preset()
        {
            Guid = Guid.NewGuid();
        }

        public override T Process(T input)
        {
            return Script != null ? input + Chain : input;
        }

        public override void Initialize()
        {
            base.Initialize();

            if (Script != null)
                Chain.Initialize();
        }

        public override void Reset()
        {
            base.Reset();

            if (Script != null)
                Chain.Reset();
        }

        #endregion

        #region ChainUi Implementation

        [YAXDontSerialize]
        public string Description
        {
            get { return Script.Descriptor.Description; }
        }

        [YAXDontSerialize]
        public Chain<T> Chain
        {
            get { return Script.Chain; }
        }

        [YAXDontSerialize]
        public string Category
        {
            get { return Script.Category; }
        }

        [YAXDontSerialize]
        public int Version
        {
            get { return Script.Version; }
        }

        [YAXDontSerialize]
        public ExtensionUiDescriptor Descriptor
        {
            get { return Script.Descriptor; }
        }

        public bool HasConfigDialog()
        {
            return Script != null && Script.HasConfigDialog();
        }

        public bool ShowConfigDialog(IWin32Window owner)
        {
            return Script != null && Script.ShowConfigDialog(owner);
        }

        public TScript CreateScript()
        {
            return (Script != null) ? Script.CreateScript() : null;
        }

        public void Destroy()
        {
            if (Script != null) Script.Destroy();
        }

        public void Dispose()
        {
            if (Script != null) Script.Dispose();
        }

        #endregion

        public override string ToString()
        {
            return Name;
        }

        public static Preset<T, TScript> Make<S>(string name = null)
            where S : IChainUi<T, TScript>, new()
        {
            var script = new S();
            return new Preset<T, TScript> { Name = (name ?? script.Descriptor.Name), Script = script };
        }
    }

    public class PresetCollection< T, TScript> : Chain<T>, INameable
        where TScript : class, IScript
    {
        #region Settings

        public List<Preset<T, TScript>> Options { get; set; }

        [YAXDontSerialize]
        public string Name { protected get; set; }

        [YAXDontSerialize]
        protected List<Preset<T, TScript>> ActiveOptions { get; private set; }

        #endregion

        public PresetCollection()
        {
            Options = new List<Preset<T, TScript>>();
        }

        public override T Process(T input)
        {
            throw new NotImplementedException();
        }

        public override void Initialize()
        {
            base.Initialize();

            var options = Options;
            if (options == null)
            {
                ActiveOptions = null;
                return;
            }

            foreach (var option in options)
            {
                option.Initialize();
            }

            ActiveOptions = options;
        }

        public override void Reset()
        {
            base.Reset();

            if (ActiveOptions == null)
                return;

            foreach (var option in ActiveOptions)
            {
                option.Reset();
            }
        }
    }
}