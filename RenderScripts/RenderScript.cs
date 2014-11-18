using System;
using System.Data;
using System.Drawing;
using System.IO;
using System.Windows.Forms;
using TransformFunc = System.Func<System.Drawing.Size, System.Drawing.Size>;

namespace Mpdn.RenderScript
{
    public class RenderScriptDescriptor
    {
        public Guid Guid = Guid.Empty;
        public string Name;
        public string Description;
        public string Copyright = "";
    }

    public abstract class RenderScript<TChain> : IScriptRenderer
        where TChain : class, IRenderChain, new()
    {
        protected IRenderer Renderer;
        protected virtual TChain Chain { get { return m_Chain; } }

        protected abstract RenderScriptDescriptor ScriptDescriptor { get; }

        #region Implementation

        private TChain m_Chain;
        private OutputFilter m_Filter;

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

        public virtual void Setup(IRenderer renderer)
        {
            Renderer = renderer;
            ShaderCache.Renderer = Renderer;
            Chain.Renderer = Renderer;
            UpdateFilter();
        }

        public virtual void Initialize(int instanceId)
        {
            m_Chain = new TChain();
        }
        

        public virtual bool ShowConfigDialog(IWin32Window owner)
        {
            throw new NotImplementedException("Config dialog has not been implemented");
        }

        public virtual ScriptInterfaceDescriptor InterfaceDescriptor
        {
            get
            {
                return new ScriptInterfaceDescriptor
                {
                    OutputSize = m_Filter.OutputSize
                };
            }
        }

        public void Render()
        {
            m_Filter.Render();
        }

        public OutputFilter Build(IRenderChain filterChain)
        {
            return new OutputFilter(Renderer, filterChain.CreateFilter(new SourceFilter(Renderer))); ;
        }

        public virtual void OnInputSizeChanged()
        {
            UpdateFilter();
        }

        public virtual void OnOutputSizeChanged()
        {
            UpdateFilter();
        }

        public virtual void UpdateFilter()
        {
            m_Filter = Build(Chain);
        }

        public void Destroy()
        {
            Common.Dispose(Chain);
            Common.Dispose(m_Filter);
        }

        #endregion Implementation
    }
}
