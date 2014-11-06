using System;
using System.Collections.Generic;
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

            public override ScriptDescriptor Descriptor
            {
                get
                {
                    return new ScriptDescriptor
                    {
                        Guid = new Guid("50CA262F-65B6-4A0F-A8B5-5E25B6A18217"),
                        Name = "Image Processor",
                        Description = "Pixel shader pre-/post-processing filter",
                        HasConfigDialog = true
                    };
                }
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

            protected override void Dispose(bool disposing)
            {
                Common.Dispose(ref m_ImageFilter);

                for (int i = 0; i < m_Shaders.Length; i++)
                {
                    Common.Dispose(ref m_Shaders[i]);
                }
            }

            public override bool ShowConfigDialog()
            {
                var dialog = new ImageProcessorConfigDialog();
                dialog.Setup(ShaderDataFilePath, m_Settings.Config);
                if (dialog.ShowDialog() != DialogResult.OK)
                    return false;

                m_Settings.Save();
                return true;
            }

            protected override ITexture GetFrame()
            {
                return m_ImageFilter == null ? InputFilter.OutputTexture : GetFrame(m_ImageFilter);
            }

            public override void OnOutputSizeChanged()
            {
                base.OnOutputSizeChanged();

                Common.Dispose(ref m_ImageFilter);
                SetupRenderChain();
            }

            private void SetupRenderChain()
            {
                var shaderFileNames = m_Settings.Config.ShaderFileNames;
                var shaders = CompileShaders(shaderFileNames);

                m_ImageFilter = shaders.Aggregate(InputFilter, (current, shader) => CreateFilter(shader, current));
            }

            private IEnumerable<IShader> CompileShaders(string[] shaderFileNames)
            {
                m_Shaders = new IShader[shaderFileNames.Length];
                for (int i = 0; i < shaderFileNames.Length; i++)
                {
                    m_Shaders[i] = CompileShader(shaderFileNames[i]);
                }
                return m_Shaders;
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
