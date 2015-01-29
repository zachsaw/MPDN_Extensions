using System;
using System.Collections.Generic;
using System.IO;
using System.Windows.Forms;
using TransformFunc = System.Func<System.Drawing.Size, System.Drawing.Size>;
using YAXLib;

namespace Mpdn.RenderScript
{
    public static class ShaderCache<T> where T: class
    {
        private static readonly Dictionary<string, ShaderWithDateTime> s_CompiledShaders =
            new Dictionary<string, ShaderWithDateTime>();

        public static T CompileShader(string shaderPath, Func<string, T> compileFunc)
        {
            var lastMod = File.GetLastWriteTimeUtc(shaderPath);

            ShaderWithDateTime result;
            if (s_CompiledShaders.TryGetValue(shaderPath, out result) &&
                result.LastModified == lastMod)
            {
                return result.Shader;
            }

            if (result != null)
            {
                Common.Dispose(result.Shader);
                s_CompiledShaders.Remove(shaderPath);
            }

            var shader = compileFunc(shaderPath);
            s_CompiledShaders.Add(shaderPath, new ShaderWithDateTime(shader, lastMod));
            return shader;
        }

        public class ShaderWithDateTime
        {
            public T Shader { get; private set; }
            public DateTime LastModified { get; private set; }

            public ShaderWithDateTime(T shader, DateTime lastModified)
            {
                Shader = shader;
                LastModified = lastModified;
            }
        }
    }

    public class RenderChainScript : IRenderScript, IDisposable
    {
        private readonly TextureCache m_Cache;
        private SourceFilter m_SourceFilter;
        private IFilter m_Filter;

        protected RenderChain Chain;

        public RenderChainScript(RenderChain chain)
        {
            Chain = chain;
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
                if (m_SourceFilter == null)
                    return null;

                return new ScriptInterfaceDescriptor
                {
                    WantYuv = true,
                    Prescale = (m_SourceFilter.LastDependentIndex > 0),
                    PrescaleSize = m_SourceFilter.OutputSize
                };
            }
        }

        public void Update()
        {
            m_SourceFilter = new SourceFilter();
            var rgbInput = m_SourceFilter.Transform(x => new RgbFilter(x));
            m_Filter = Chain.CreateFilter(rgbInput);
            m_Filter = m_Filter.Initialize();
        }

        public void Render()
        {
            m_Cache.PutTempTexture(Renderer.OutputRenderTarget);
            m_Filter.Render(m_Cache);
            if (Renderer.OutputRenderTarget != m_Filter.OutputTexture)
            {
                Scale(Renderer.OutputRenderTarget, m_Filter.OutputTexture);
            }
            m_Filter.Reset(m_Cache);
            m_Cache.FlushTextures();
        }

        private static void Scale(ITexture output, ITexture input)
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
        RenderChain GetChain();
    }

    public abstract class RenderChainUi<TChain> : IRenderChainUi
        where TChain : RenderChain, new()
    {
        [YAXSerializeAs("Settings")]
        public virtual TChain Chain { get; set; }

        protected abstract RenderScriptDescriptor ScriptDescriptor { get; }

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

        public virtual void Initialize()
        {
            Chain = new TChain();
        }

        public RenderChain GetChain()
        {
            return Chain;
        }

        [YAXDontSerialize]
        public virtual ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = ScriptDescriptor.Guid,
                    Name = ScriptDescriptor.Name,
                    Description = ScriptDescriptor.Description,
                    Copyright = ScriptDescriptor.Copyright
                };
            }
        }

        public virtual bool HasConfigDialog()
        {
            return false;
        }

        public virtual bool ShowConfigDialog(IWin32Window owner)
        {
            throw new NotImplementedException("Config dialog has not been implemented");
        }

        public virtual void Destroy()
        {
            Common.Dispose(m_RenderScript);
        }

        #endregion Implementation
    }
}