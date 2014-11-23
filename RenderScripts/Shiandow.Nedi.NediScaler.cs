using System;
using System.Drawing;
using System.Windows.Forms;
using YAXLib;

namespace Mpdn.RenderScript
{
    namespace Shiandow.Nedi
    {
        public class Nedi : RenderChain
        {
            #region Settings

            public Nedi()
            {
                AlwaysDoubleImage = false;
            }

            [YAXErrorIfMissed(YAXExceptionTypes.Ignore)]
            public bool AlwaysDoubleImage { get; set; }

            #endregion

            private bool UseNedi(IFilter SourceFilter)
            {
                var size = SourceFilter.OutputSize;
                if (size.IsEmpty)
                    return false;

                if (AlwaysDoubleImage)
                    return true;

                return Renderer.TargetSize.Width > size.Width ||
                       Renderer.TargetSize.Height > size.Height;
            }

            public override IFilter CreateFilter(IFilter SourceFilter)
            {
                var m_Nedi1Shader = CompileShader("NEDI-I.hlsl");
                var m_Nedi2Shader = CompileShader("NEDI-II.hlsl");
                var m_NediHInterleaveShader = CompileShader("NEDI-HInterleave.hlsl");
                var m_NediVInterleaveShader = CompileShader("NEDI-VInterleave.hlsl");

                Func<Size, Size> transformWidth  = s => new Size(2 * s.Width, s.Height);
                Func<Size, Size> transformHeight = s => new Size(s.Width, 2 * s.Height);

                if (!UseNedi(SourceFilter))
                    return SourceFilter;

                var nedi1 = new ShaderFilter(m_Nedi1Shader, SourceFilter);
                var nediH = new ShaderFilter(m_NediHInterleaveShader, transformWidth, SourceFilter, nedi1);
                var nedi2 = new ShaderFilter(m_Nedi2Shader, nediH);
                var nediV = new ShaderFilter(m_NediVInterleaveShader, transformHeight, nediH, nedi2);

                return nediV;
            }
        }

        public class NediScaler : ConfigurableRenderChainUi<Nedi, NediConfigDialog>
        {
            protected override string ConfigFileName
            {
                get { return "Shiandow.Nedi"; }
            }

            protected override RenderScriptDescriptor ScriptDescriptor
            {
                get
                {
                    return new RenderScriptDescriptor
                    {
                        Guid = new Guid("B8E439B7-7DC2-4FC1-94E2-608A39756FB0"),
                        Name = "NEDI",
                        Description = GetDescription(),
                        Copyright = "NEDI by Shiandow",
                    };
                }
            }

            private string GetDescription()
            {
                var options = ScriptConfig == null
                    ? string.Empty
                    : string.Format("{0}", ScriptConfig.Config.AlwaysDoubleImage ? " (forced)" : string.Empty);
                return string.Format("NEDI image doubler{0}", options);
            }
        }
    }
}
