using System;
using System.Collections.Generic;
using System.Reflection;
using System.Linq;
using YAXLib;

namespace Mpdn.RenderScript
{
    namespace Mpdn.ScriptChain
    {
        public class ChainUiPair
        {
            [YAXErrorIfMissed(YAXExceptionTypes.Ignore)]
            public IRenderChain Chain { get; set; }
            [YAXErrorIfMissed(YAXExceptionTypes.Ignore)]
            public Type UiType { get; set; }

            public IRenderChainUi ChainUi { 
                get 
                { 
                    if (m_ChainUi == null)
                        m_ChainUi = CreateUi();
                    return m_ChainUi;
                }
                protected set
                {
                    m_ChainUi = ChainUi;
                }
            }
            
            private IRenderChainUi m_ChainUi;
            private IRenderChainUi CreateUi() {
                var Ui = (IRenderChainUi)UiType.GetConstructor(Type.EmptyTypes).Invoke(new object[0]);
                Ui.Initialize();
                Ui.SetChain(Chain);
                return Ui;
            }
        }

        public class ScriptChain : CombinedChain
        {
            public ScriptChain()
            {
                ScriptList = new List<ChainUiPair>();
            }

            [YAXErrorIfMissed(YAXExceptionTypes.Ignore)]
            public IList<ChainUiPair> ScriptList { get; set; }

            protected override void BuildChain(FilterChain Chain)
            {
                foreach (var pair in ScriptList)
                    Chain.Add(pair.Chain);
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
