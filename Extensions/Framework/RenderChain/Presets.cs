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
using SharpDX;

namespace Mpdn.Extensions.Framework.RenderChain
{
    public interface INameable
    {
        string Name { set; }
    }

    public class Preset : RenderChain
    {
        #region Settings

        private string m_Name;
        private IRenderChainUi m_Script;

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

        public IRenderChainUi Script
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

        #region Script Properties

        [YAXDontSerialize]
        public string Description
        {
            get { return Script.Descriptor.Description; }
        }

        [YAXDontSerialize]
        public RenderChain Chain
        {
            get { return Script.Chain; }
        }

        public bool HasConfigDialog()
        {
            return Script != null && Script.HasConfigDialog();
        }

        public bool ShowConfigDialog(IWin32Window owner)
        {
            return Script != null && Script.ShowConfigDialog(owner);
        }

        #endregion

        #region RenderChain implementation

        public Preset()
        {
            Guid = Guid.NewGuid();
        }

        protected override IFilter CreateFilter(IFilter input)
        {
            return Script != null ? input + Chain : input;
        }

        public override void Initialize()
        {
            base.Initialize();

            if (Script == null)
                return;

            Chain.Initialize();
        }

        public override void Reset()
        {
            base.Reset();

            if (Script != null)
                Chain.Reset();
        }

        public override Func<string> Status
        {
            get { return Script != null ? Chain.Status : Inactive; }
            protected set { throw new NotImplementedException(); }
        }

        public override void MarkInactive()
        {
            if (Script != null)
                Chain.MarkInactive();
        }

        #endregion

        public override string ToString()
        {
            return Name;
        }

        public static Preset Make<T>(string name = null)
            where T : IRenderChainUi, new()
        {
            var script = new T();
            return new Preset { Name = (name ?? script.Descriptor.Name), Script = script };
        }
    }

    public static class PresetHelper
    {
        public static Preset MakeNewPreset(this IRenderChainUi renderScript, string name = null)
        {
            return renderScript.CreateNew().ToPreset();
        }

        public static Preset ToPreset(this IRenderChainUi renderScript, string name = null)
        {
            return new Preset { Name = name ?? renderScript.Descriptor.Name, Script = renderScript };
        }
    }

    public class PresetCollection : RenderChain, INameable
    {
        #region Settings

        public List<Preset> Options { get; set; }

        [YAXDontSerialize]
        public virtual bool AllowRegrouping
        {
            get { return true; }
        }

        [YAXDontSerialize]
        public string Name { protected get; set; }

        #endregion

        public PresetCollection()
        {
            Options = new List<Preset>();
        }

        protected override IFilter CreateFilter(IFilter input)
        {
            throw new NotImplementedException();
        }

        public override void Initialize()
        {
            base.Initialize();

            if (Options == null)
                return;

            foreach (var option in Options)
            {
                option.Initialize();
            }
        }

        public override void Reset()
        {
            base.Reset();

            if (Options == null)
                return;

            foreach (var option in Options)
            {
                option.Reset();
            }
        }
    }

    public class ChromaScalerPreset : Preset, IChromaScaler
    {
        public IFilter CreateChromaFilter(IFilter lumaInput, IFilter chromaInput, TextureSize targetSize, Vector2 chromaOffset)
        {
            IChromaScaler chromaScaler = Chain as IChromaScaler ?? new DefaultChromaScaler();

            return chromaScaler.CreateChromaFilter(lumaInput, chromaInput, targetSize, chromaOffset);
        }
    }

    public static class ChromaScalerPresetHelper
    {
        public static ChromaScalerPreset MakeNewChromaScalerPreset(this IRenderChainUi renderScript, string name = null)
        {
            return renderScript.CreateNew().ToChromaScalerPreset();
        }

        public static ChromaScalerPreset ToChromaScalerPreset(this IRenderChainUi renderScript, string name = null)
        {
            return new ChromaScalerPreset { Name = name ?? renderScript.Descriptor.Name, Script = renderScript };
        }
    }
}