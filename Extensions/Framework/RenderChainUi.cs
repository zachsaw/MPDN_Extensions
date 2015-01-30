using System;
using Mpdn.Config;
using YAXLib;

namespace Mpdn.RenderScript
{
    public interface IRenderChainUi : IRenderScriptUi
    {
        RenderChain GetChain();
    }

    public abstract class RenderChainUi<TChain> : RenderChainUi<TChain, ScriptConfigDialog<TChain>>
        where TChain : RenderChain, new()
    { }

    public abstract class RenderChainUi<TChain, TDialog> : ExtensionUi<Config.Internal.RenderScripts, TChain, TDialog>, IRenderChainUi
        where TChain : RenderChain, new()
        where TDialog : ScriptConfigDialog<TChain>, new()
    {
        [YAXSerializeAs("Settings")]
        public TChain Chain
        {
            get { return ScriptConfig.Config; }
            set { ScriptConfig = new Config(value); }
        }

        public IRenderScript CreateRenderScript()
        {
            return m_RenderScript ?? (m_RenderScript = new RenderChainScript(Chain));
        }

        #region Implementation

        private RenderChainScript m_RenderScript;

        [YAXDontSerialize]
        public virtual ScriptInterfaceDescriptor InterfaceDescriptor
        {
            get { return CreateRenderScript().Descriptor; }
        }

        public override void Initialize()
        {
            base.Initialize();
        }

        public RenderChain GetChain()
        {
            return Chain;
        }

        public override void Destroy()
        {
            DisposeHelper.Dispose(m_RenderScript);
            base.Destroy();
        }

        #endregion Implementation
    }
}