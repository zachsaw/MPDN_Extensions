using System;
using System.Drawing;
using System.Windows.Forms;
using YAXLib;

namespace Mpdn.RenderScript
{
    namespace Shiandow.Nedi
    {
        #region Settings

        public class Settings
        {
            public Settings()
            {
                AlwaysDoubleImage = false;
            }

            [YAXErrorIfMissed(YAXExceptionTypes.Ignore)]
            public bool AlwaysDoubleImage { get; set; }
        }

        #endregion

        public class NediScaler : ConfigurableRenderScript<Settings, NediConfigDialog>
        {
            private IShader m_Nedi1Shader;
            private IShader m_Nedi2Shader;
            private IShader m_NediHInterleaveShader;
            private IFilter m_NediScaler;
            private IShader m_NediVInterleaveShader;

            public static NediScaler Create(bool forced = false)
            {
                var result = new NediScaler();
                result.Initialize();
                result.Settings.Config.AlwaysDoubleImage = forced;
                return result;
            }

            protected override string ShaderPath
            {
                get { return "NEDI"; }
            }

            protected override ConfigurableRenderScriptDescriptor ConfigScriptDescriptor
            {
                get
                {
                    return new ConfigurableRenderScriptDescriptor
                    {
                        Guid = new Guid("B8E439B7-7DC2-4FC1-94E2-608A39756FB0"),
                        Name = "NEDI",
                        Description = GetDescription(),
                        Copyright = "NEDI by Shiandow",
                        ConfigFileName = "Shiandow.Nedi"
                    };
                }
            }

            public override ScriptInterfaceDescriptor InterfaceDescriptor
            {
                get
                {
                    var videoSize = Renderer.VideoSize;
                    var outputSize = UseNedi ? new Size(videoSize.Width*2, videoSize.Height*2) : videoSize;
                    return new ScriptInterfaceDescriptor
                    {
                        OutputSize = outputSize
                    };
                }
            }

            protected override TextureAllocTrigger TextureAllocTrigger
            {
                get { return TextureAllocTrigger.OnInputOutputSizeChanged; }
            }

            public override IFilter CreateFilter(Settings settings)
            {
                lock (Settings)
                {
                    CompileShaders();
                    return CreateNediScaler();
                }
            }

            public override IFilter GetFilter()
            {
                lock (Settings)
                {
                    return UseNedi ? m_NediScaler : SourceFilter;
                }
            }

            public override bool ShowConfigDialog(IWin32Window owner)
            {
                if (!base.ShowConfigDialog(owner)) 
                    return false;

                OnInputSizeChanged();
                return true;
            }

            protected override void Dispose(bool disposing)
            {
                DiscardNediScaler();
                DiscardNediShaders();
            }

            private bool UseNedi
            {
                get { return Settings.Config.AlwaysDoubleImage || NeedToUpscale(); }
            }

            private string GetDescription()
            {
                var options = Settings == null
                    ? string.Empty
                    : string.Format("{0}", Settings.Config.AlwaysDoubleImage ? " (forced)" : string.Empty);
                return string.Format("NEDI image doubler{0}", options);
            }

            private void DiscardNediShaders()
            {
                Common.Dispose(ref m_Nedi1Shader);
                Common.Dispose(ref m_Nedi2Shader);
                Common.Dispose(ref m_NediHInterleaveShader);
                Common.Dispose(ref m_NediVInterleaveShader);
            }

            private void CompileShaders()
            {
                m_Nedi1Shader = CompileShader("NEDI-I.hlsl");
                m_Nedi2Shader = CompileShader("NEDI-II.hlsl");
                m_NediHInterleaveShader = CompileShader("NEDI-HInterleave.hlsl");
                m_NediVInterleaveShader = CompileShader("NEDI-VInterleave.hlsl");
            }

            private IFilter CreateNediScaler()
            {
                Func<Size, Size> transformWidth = s => new Size(2*s.Width, s.Height);
                Func<Size, Size> transformHeight = s => new Size(s.Width, s.Height*2);

                var nedi1 = CreateFilter(m_Nedi1Shader, SourceFilter);
                var nediH = CreateFilter(m_NediHInterleaveShader, transformWidth, SourceFilter, nedi1);
                var nedi2 = CreateFilter(m_Nedi2Shader, nediH);
                var nediV = CreateFilter(m_NediVInterleaveShader, transformHeight, nediH, nedi2);

                return m_NediScaler = nediV;
            }

            private bool NeedToUpscale()
            {
                return Renderer.TargetSize.Width > Renderer.VideoSize.Width ||
                       Renderer.TargetSize.Height > Renderer.VideoSize.Height;
            }

            private void DiscardNediScaler()
            {
                // Disposes the all ancestors too 
                Common.Dispose(ref m_NediScaler);
            }
        }
    }
}
