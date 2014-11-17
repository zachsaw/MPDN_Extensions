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

        public class NediChain : ChainBuilder<Settings>
        {
            protected override string ShaderPath
            {
                get { return "NEDI"; }
            }

            private bool NeedToUpscale()
            {
                return Renderer.TargetSize.Width > Renderer.VideoSize.Width ||
                       Renderer.TargetSize.Height > Renderer.VideoSize.Height;
            }

            protected override IFilter CreateFilter(IFilter SourceFilter, Settings settings)
            {
                var m_Nedi1Shader = CompileShader("NEDI-I.hlsl");
                var m_Nedi2Shader = CompileShader("NEDI-II.hlsl");
                var m_NediHInterleaveShader = CompileShader("NEDI-HInterleave.hlsl");
                var m_NediVInterleaveShader = CompileShader("NEDI-VInterleave.hlsl");

                Func<Size, Size> transformWidth = s => new Size(2 * s.Width, s.Height);
                Func<Size, Size> transformHeight = s => new Size(s.Width, s.Height * 2);

                var nedi1 = CreateFilter(m_Nedi1Shader, SourceFilter);
                var nediH = CreateFilter(m_NediHInterleaveShader, transformWidth, SourceFilter, nedi1);
                var nedi2 = CreateFilter(m_Nedi2Shader, nediH);
                var nediV = CreateFilter(m_NediVInterleaveShader, transformHeight, nediH, nedi2);

                Func<bool> UseNedi = () => settings.AlwaysDoubleImage || NeedToUpscale();

                return new IfElseFilter(UseNedi, nediV, SourceFilter);
            }
        }

        public class NediScaler : ConfigurableRenderScript<Settings, NediChain, NediConfigDialog>
        {
            public static NediScaler Create(bool forced = true)
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

            protected override TextureAllocTrigger TextureAllocTrigger
            {
                get { return TextureAllocTrigger.OnInputOutputSizeChanged; }
            }

            public override bool ShowConfigDialog(IWin32Window owner)
            {
                if (!base.ShowConfigDialog(owner)) 
                    return false;

                OnInputSizeChanged();
                return true;
            }

            private string GetDescription()
            {
                var options = Settings == null
                    ? string.Empty
                    : string.Format("{0}", Settings.Config.AlwaysDoubleImage ? " (forced)" : string.Empty);
                return string.Format("NEDI image doubler{0}", options);
            }
        }
    }
}
