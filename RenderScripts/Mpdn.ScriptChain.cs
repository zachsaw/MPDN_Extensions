using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using YAXLib;

namespace Mpdn.RenderScript
{
    namespace Mpdn.ScriptChain
    {
        public class ScriptChain : RenderChain
        {
            private List<IRenderChainUi> m_ScriptList;
            public List<IRenderChainUi> ScriptList 
            {
                get { return m_ScriptList; }

                set
                {
                    m_ScriptList = value;
                    foreach (var script in m_ScriptList) script.Initialize();
                }
            }

            public ScriptChain()
            {
                ScriptList = new List<IRenderChainUi>();
            }

            public override IFilter CreateFilter(IResizeableFilter sourceFilter)
            {
                return ScriptList.Select(pair => pair.GetChain()).Aggregate(sourceFilter, (a, b) => a + b);
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
                    : string.Join(" ➔ ", Chain.ScriptList.Select(x => x.Descriptor.Name));
            }
        }
    }
}