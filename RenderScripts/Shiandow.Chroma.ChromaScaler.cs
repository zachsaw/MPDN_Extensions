using System;
using System.ComponentModel;
using System.Drawing;
using System.Collections.Generic;
using SharpDX;
using TransformFunc = System.Func<System.Drawing.Size, System.Drawing.Size>;
using YAXLib;

namespace Mpdn.RenderScript
{
    namespace Shiandow.Chroma
    {
        #region Filter Definitions

        public class ChromaFilter : ShaderFilter
        {
            private float B;
            private float C;

            public ChromaFilter(IRenderer renderer, IShader shader, float B, float C, TransformFunc transform, params IFilter[] inputFilters)
                : base(renderer, shader, transform, 0, false, inputFilters)
            { 
                this.B = B;
                this.C = C;
            }

            protected override void LoadInputs(IEnumerable<ITexture> inputs)
            {
                base.LoadInputs(inputs);
                Shader.SetConstant("BC", new Vector4(B, C, 0, 0), false);
            }
        }

        #endregion

        #region Settings

        public class Settings
        {
            public Settings()
            {
                B = (float)(1.0 / 3.0);
                C = (float)(1.0 / 3.0);
                Preset = Presets.Custom;
            }

            [YAXErrorIfMissed(YAXExceptionTypes.Ignore)]
            public float B { get; set; }

            [YAXErrorIfMissed(YAXExceptionTypes.Ignore)]
            public float C { get; set; }

            [YAXErrorIfMissed(YAXExceptionTypes.Ignore)]
            public Presets Preset { get; set; }
        }

        #endregion

        #region Presets

        public enum Presets
        {
            [Description("Custom")]
            Custom = -1,
            [Description("Hermite")]
            Hermite = 0,
            [Description("Spline")]
            Spline = 1,
            [Description("Catmull-Rom")]
            CatmullRom = 2,
            [Description("Mitchell-Netravali")]
            MitchellNetravali = 3,
            [Description("Robidoux")]
            Robidoux = 4,
            [Description("Robidoux-Sharp")]
            RobidouxSharp = 5,
            [Description("Robidoux-Soft")]
            RobidouxSoft = 6
        }

        #endregion

        public class ChromaScaler : ConfigurableRenderScript<Settings, ChromaScalerConfigDialog>
        {
            public static readonly double[] B_CONST = {0.0, 1.0,     0.0, 1.0/3.0,  12/(19+  9*Math.Sqrt(2)), 6/(13+7*Math.Sqrt(2)), ( 9-3*Math.Sqrt(2))/7 };
            public static readonly double[] C_CONST = {0.0, 0.0, 1.0/2.0, 1.0/3.0, 113/(58+216*Math.Sqrt(2)), 7/(2+12*Math.Sqrt(2)), (-2+3*Math.Sqrt(2))/14};

            private IFilter m_ChromaScaler;
        	private IShader m_ChromaShader;

            public static ChromaScaler Create(Presets preset)
            {
                if (preset == Presets.Custom)
                {
                    throw new ArgumentException(
                        "To use custom preset, call the Create() overload with B and C arguments", "preset");
                }

                return Create((float) B_CONST[(int) preset], (float) C_CONST[(int) preset]);
            }

            public static ChromaScaler Create(float B = (float) (1.0/3.0), float C = (float) (1.0/3.0))
            {
                var result = new ChromaScaler();
                result.Initialize();

                result.Settings.Config.B = B;
                result.Settings.Config.C = C;
                return result;
            }

            protected override string ShaderPath
            {
                get { return "ChromaScaler"; }
            }

            protected override ConfigurableRenderScriptDescriptor ConfigScriptDescriptor
            {
                get
                {
                    return new ConfigurableRenderScriptDescriptor
                    {
                        Guid = new Guid("BDCC94DD-93B3-4414-BA1F-345E10E1C371"),
                        Name = "ChromaScaler",
                        Description = "Chroma Scaler",
                        Copyright = "ChromaScaler by Shiandow",
                        ConfigFileName = "Shiandow.Chroma"
                    };
                }
            }

            public override ScriptInterfaceDescriptor InterfaceDescriptor
            {
                get
                {
                    return new ScriptInterfaceDescriptor
                    {
                        Prescale = false
                    };
                }
            }

            protected override TextureAllocTrigger TextureAllocTrigger
            {
                get { return TextureAllocTrigger.OnInputOutputSizeChanged; }
            }

            public override IFilter CreateFilter(Settings settings)
            {
                m_ChromaShader = CompileShader("Chroma.hlsl");

                Func<Size, Size> transformWidth = s => new Size(Renderer.VideoSize.Width, s.Height);
                Func<Size, Size> transformHeight = s => new Size(s.Width, Renderer.VideoSize.Height);
                Func<Size, Size> transformWidthHeight = s => Renderer.VideoSize;

                var Yinput = new YSourceFilter(Renderer);
                var Uinput = new USourceFilter(Renderer);
                var Vinput = new VSourceFilter(Renderer);

                var chroma = new ChromaFilter(Renderer, m_ChromaShader, settings.B, settings.C, transformWidthHeight, Yinput, Uinput, Vinput);
                var rgb = new RgbFilter(Renderer, chroma);

                return m_ChromaScaler = rgb;
            }

            protected override void Dispose(bool disposing)
            {
                base.Dispose(disposing);

                Common.Dispose(ref m_ChromaScaler);
                Common.Dispose(ref m_ChromaShader);
            }
        }
    }
}
