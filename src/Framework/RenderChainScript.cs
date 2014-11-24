using System;
using System.Collections.Generic;
using System.Windows.Forms;
using TransformFunc = System.Func<System.Drawing.Size, System.Drawing.Size>;

namespace Mpdn.RenderScript
{
    public static class ShaderCache
    {
        private static readonly Dictionary<String, IShader> s_CompiledShaders = new Dictionary<string, IShader>();

        public static IShader CompileShader(String shaderPath)
        {
            IShader shader;
            s_CompiledShaders.TryGetValue(shaderPath, out shader);

            if (shader == null)
            {
                shader = Renderer.CompileShader(shaderPath);
                s_CompiledShaders.Add(shaderPath, shader);
            }

            return shader;
        }
    }

    public class RenderChainScript : IRenderScript, IDisposable
    {
        private readonly TextureCache m_Cache;
        private readonly SourceFilter m_SourceFilter;
        protected IRenderChain Chain;
        private IFilter m_Filter;

        public RenderChainScript(IRenderChain chain)
        {
            Chain = chain;
            m_SourceFilter = new SourceFilter();
            m_Cache = new TextureCache();
        }

        public void Dispose()
        {
            Common.Dispose(m_Cache);
        }

        public ScriptInterfaceDescriptor Descriptor
        {
            get
            {
                return new ScriptInterfaceDescriptor
                {
                    WantYuv = m_SourceFilter.WantYuv,
                    Prescale = (m_SourceFilter.LastDependentIndex > 0),
                    PrescaleSize = m_SourceFilter.GetOutputSize(false)
                };
            }
        }

        public void Update()
        {
            m_Filter = Chain.CreateFilter(m_SourceFilter);
            m_Filter.Initialize();
        }

        public void Render()
        {
            m_Cache.PutTempTexture(Renderer.OutputRenderTarget);
            m_Filter.NewFrame();
            m_Filter.Render(m_Cache);
            Scale(Renderer.OutputRenderTarget, m_Filter.OutputTexture);
            m_Filter.ReleaseTexture(m_Cache);
            m_Cache.FlushTextures();
        }

        private void Scale(ITexture output, ITexture input)
        {
            Renderer.Scale(output, input, Renderer.LumaUpscaler, Renderer.LumaDownscaler);
        }
    }

    public class RenderScriptDescriptor
    {
        public string Copyright = "";
        public string Description;
        public Guid Guid = Guid.Empty;
        public string Name;
    }

    public interface IRenderChainUi : IRenderScriptUi
    {
        IRenderChain GetChain();
        void Initialize(IRenderChain renderChain);
    }

    public abstract class RenderChainUi<TChain> : IRenderChainUi
        where TChain : class, IRenderChain, new()
    {
        protected virtual TChain Chain
        {
            get { return m_Chain; }
        }

        protected abstract RenderScriptDescriptor ScriptDescriptor { get; }

        public IRenderScript RenderScript
        {
            get { return m_RenderScript ?? (m_RenderScript = new RenderChainScript(Chain)); }
        }

        #region Implementation

        private TChain m_Chain;
        private IRenderScript m_RenderScript;

        public virtual ScriptInterfaceDescriptor InterfaceDescriptor
        {
            get { return RenderScript.Descriptor; }
        }

        public virtual void Initialize()
        {
            m_Chain = new TChain();
        }

        public virtual void Initialize(IRenderChain renderChain)
        {
            m_Chain = renderChain as TChain ?? new TChain();
        }

        public IRenderChain GetChain()
        {
            return Chain;
        }

        public virtual ScriptDescriptor Descriptor
        {
            get
            {
                return new ScriptDescriptor
                {
                    HasConfigDialog = false,
                    Guid = ScriptDescriptor.Guid,
                    Name = ScriptDescriptor.Name,
                    Description = ScriptDescriptor.Description,
                    Copyright = ScriptDescriptor.Copyright
                };
            }
        }

        public virtual bool ShowConfigDialog(IWin32Window owner)
        {
            throw new NotImplementedException("Config dialog has not been implemented");
        }

        public void Destroy()
        {
            Common.Dispose(m_RenderScript);
        }

        #endregion Implementation
    }
}