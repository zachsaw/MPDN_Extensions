using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using YAXLib;

namespace Mpdn.RenderScript
{
    namespace Mpdn.ScriptChain
    {
        public class ChainUiPair
        {
            private IRenderChainUi m_ChainUi;

            public ChainUiPair()
            {
            }

            public ChainUiPair(IRenderChainUi scripUi)
            {
                Chain = scripUi.GetChain();
                m_ChainUi = scripUi;
                UiType = scripUi.GetType();
            }

            [YAXSerializeAs("RenderChain")]
            public IRenderChain Chain { get; set; }

            [YAXDontSerialize]
            public Type UiType { get; set; }

            [YAXDontSerialize]
            public IRenderChainUi ChainUi
            {
                get { return m_ChainUi ?? (m_ChainUi = CreateUi()); }
            }

            [YAXSerializeAs("RenderChainUi")]
            [YAXErrorIfMissed(YAXExceptionTypes.Ignore)]
            public String UiTypeName
            {
                get { return UiType.FullName; }
                set { UiType = Assembly.GetExecutingAssembly().GetType(value, false); }
            }

            private IRenderChainUi CreateUi()
            {
                var constructor = UiType.GetConstructor(Type.EmptyTypes);
                if (constructor == null)
                {
                    throw new EntryPointNotFoundException("RenderChainUi must implement parameter-less constructor");
                }

                var ui = (IRenderChainUi) constructor.Invoke(new object[0]);
                ui.Initialize(Chain);
                return ui;
            }
        }

        public class ScriptChain : CombinedChain
        {
            public ScriptChain()
            {
                ScriptList = new List<ChainUiPair>();
            }

            [YAXErrorIfMissed(YAXExceptionTypes.Ignore)]
            public List<ChainUiPair> ScriptList { get; set; }

            protected override void BuildChain(FilterChain chain)
            {
                foreach (var pair in ScriptList)
                {
                    chain.Add(pair.Chain);
                }
            }
        }


        public class ScriptChainScript : ConfigurableRenderChainUi<ScriptChain, ScriptChainDialog>
        {
            protected override string ConfigFileName
            {
                get { return "Mpdn.ScriptChain"; }
            }

            protected override RenderScriptDescriptor ScriptDescriptor
            {
                get
                {
                    return new RenderScriptDescriptor
                    {
                        Guid = new Guid("3A462015-2D92-43AC-B559-396DACF896C3"),
                        Name = "Script Chain",
                        Description = GetDescription(),
                    };
                }
            }

            private string GetDescription()
            {
                return ScriptConfig == null || Chain.ScriptList.Count == 0
                    ? "Chain of render scripts"
                    : string.Join(" ➔ ", Chain.ScriptList.Select(x => x.ChainUi.Descriptor.Name));
            }
        }
    }
}