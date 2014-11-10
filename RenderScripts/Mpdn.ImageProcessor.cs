using System;
using System.Linq;
using System.Windows.Forms;
using YAXLib;

namespace Mpdn.RenderScript
{
    namespace Mpdn.ImageProcessor
    {
        public class ImageProcessor : RenderScript
        {
            private ImageProcessorSettings m_Settings;

            private IFilter m_ImageFilter;
            private IShader[] m_Shaders = new IShader[0];
            private string[] m_ShaderFileNames = new string[0];

            public override ScriptDescriptor Descriptor
            {
                get
                {
                    return new ScriptDescriptor
                    {
                        Guid = new Guid("50CA262F-65B6-4A0F-A8B5-5E25B6A18217"),
                        Name = "Image Processor",
                        Description = GetDescription(),
                        HasConfigDialog = true
                    };
                }
            }

            private string GetDescription()
            {
                return m_Settings == null || m_Settings.Config.ShaderFileNames.Length == 0
                    ? "Pixel shader pre-/post-processing filter"
                    : string.Join(" ➔ ", m_Settings.Config.ShaderFileNames);
            }

            protected override string ShaderPath
            {
                get { return "ImageProcessingShaders"; }
            }

            public override void Initialize(int instanceId)
            {
                m_Settings = new ImageProcessorSettings(instanceId);
                m_Settings.Load();
            }

            public override void Destroy()
            {
                m_Settings.Destroy();
            }

            public override void Setup(IRenderer renderer)
            {
                base.Setup(renderer);

                SetupRenderChain();
            }

            protected override void Dispose(bool disposing)
            {
                Common.Dispose(ref m_ImageFilter);

                DisposeShaders();
            }

            public override bool ShowConfigDialog(IWin32Window owner)
            {
                using (var dialog = new ImageProcessorConfigDialog())
                {
                    dialog.Setup(ShaderDataFilePath, m_Settings.Config);
                    if (dialog.ShowDialog(owner) != DialogResult.OK)
                        return false;

                    m_Settings.Save();
                    SetupRenderChain();
                    OnOutputSizeChanged();
                    return true;
                }
            }

            protected override ITexture GetFrame()
            {
                lock (m_Settings)
                {
                    return GetFrame(m_ImageFilter);
                }
            }

            public override void OnOutputSizeChanged()
            {
                lock (m_Settings)
                {
                    if (Renderer == null)
                        return;

                    m_ImageFilter.AllocateTextures();
                }
            }

            private void SetupRenderChain()
            {
                lock (m_Settings)
                {
                    if (Renderer == null)
                        return;

                    var shaderFileNames = m_Settings.Config.ShaderFileNames;
                    if (!shaderFileNames.SequenceEqual(m_ShaderFileNames))
                    {
                        CompileShaders(shaderFileNames);
                    }

                    m_ImageFilter = m_Shaders.Aggregate(SourceFilter, (current, shader) => CreateFilter(shader, current));
                    m_ImageFilter.Initialize();
                }
            }

            private void CompileShaders(string[] shaderFileNames)
            {
                DisposeShaders();

                m_Shaders = new IShader[shaderFileNames.Length];
                for (int i = 0; i < shaderFileNames.Length; i++)
                {
                    m_Shaders[i] = CompileShader(shaderFileNames[i]);
                }
                m_ShaderFileNames = shaderFileNames;
            }

            private void DisposeShaders()
            {
                for (int i = 0; i < m_Shaders.Length; i++)
                {
                    Common.Dispose(ref m_Shaders[i]);
                }
                m_Shaders = new IShader[0];
                m_ShaderFileNames = new string[0];
            }
        }

        #region Settings

        public class Settings
        {
            private string[] m_ShaderFileNames;

            [YAXErrorIfMissed(YAXExceptionTypes.Ignore)]
            public string[] ShaderFileNames
            {
                get { return m_ShaderFileNames ?? (m_ShaderFileNames = new string[0]); }
                set { m_ShaderFileNames = value; }
            }
        }

        public class ImageProcessorSettings : ScriptSettings<Settings>
        {
            private readonly int m_InstanceId;

            public ImageProcessorSettings(int instanceId)
            {
                m_InstanceId = instanceId;
            }

            protected override string ScriptConfigFileName
            {
                get { return string.Format("Mpdn.ImageProcessor.{0}.config", m_InstanceId); }
            }
        }

        #endregion
    }
}
