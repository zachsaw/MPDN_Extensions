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
        protected OutputFilter Filter
        {
            get
            {
                if (m_Filter == null) 
                    m_Filter = new OutputFilter(Renderer, Chain);

                return m_Filter;
            }
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

        public virtual void Setup(IRenderer renderer)
        {
            Renderer = renderer;
            StaticRenderer.Renderer = Renderer;
            Chain.Renderer = Renderer;
            RefreshFilter();
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
                return Filter.Descriptor;
            }
        }

        public void Render()
        {
            Filter.Render();
        }

        public virtual void OnInputSizeChanged()
        {
            RefreshFilter();
        }

        public virtual void OnOutputSizeChanged()
        {
            RefreshFilter();
        }

        public virtual void RefreshFilter()
        {
            Filter.Refresh();
        }

        public void Destroy()
        {
            Common.Dispose(Chain);
            Common.Dispose(Filter);
        }

        #endregion Implementation
    }
}
