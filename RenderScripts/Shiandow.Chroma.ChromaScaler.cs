using System;
using System.ComponentModel;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using System.Collections.Generic;
using SharpDX;
using TransformFunc = System.Func<System.Drawing.Size, System.Drawing.Size>;
using YAXLib;

namespace Mpdn.RenderScript
{
    namespace Shiandow.Chroma
    {
	    public class YSourceFilter : BaseSourceFilter
	    {
	    	public YSourceFilter(IRenderer renderer) : base(renderer) { }

			public override ITexture OutputTexture
	        {
	            get { return Renderer.TextureY; }
	        }

	        public override Size OutputSize
	        {
	            get { return Renderer.LumaSize; }
	        }
	    }

		public class USourceFilter : BaseSourceFilter
	    {
	    	public USourceFilter(IRenderer renderer) : base(renderer) { }

			public override ITexture OutputTexture
	        {
	            get { return Renderer.TextureU; }
	        }

	        public override Size OutputSize
	        {
	            get { return Renderer.ChromaSize; }
	        }
	    }

		public class VSourceFilter : BaseSourceFilter
	    {
	    	public VSourceFilter(IRenderer renderer) : base(renderer) { }

			public override ITexture OutputTexture
	        {
	            get { return Renderer.TextureV; }
	        }

	        public override Size OutputSize
	        {
	            get { return Renderer.ChromaSize; }
	        }
	    }

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

        public class RGBFilter : Filter
        {
            public RGBFilter(IRenderer renderer, IFilter inputFilter)
                : base(renderer, inputFilter)
            { }

            public override Size OutputSize
            {
                get { return InputFilters[0].OutputSize; }
            }

            public override void Render(IEnumerable<ITexture> inputs)
            {
                Renderer.ConvertToRgb(OutputTexture, inputs.Single(), Renderer.Colorimetric);
            }
        }

        public class ChromaScaler : RenderScript
        {
            public static readonly double[] BConst = {0.0, 1.0,     0.0, 1.0/3.0,  12/(19+  9*Math.Sqrt(2)), 6/(13+7*Math.Sqrt(2)), ( 9-3*Math.Sqrt(2))/7 };
            public static readonly double[] CConst = {0.0, 0.0, 1.0/2.0, 1.0/3.0, 113/(58+216*Math.Sqrt(2)), 7/(2+12*Math.Sqrt(2)), (-2+3*Math.Sqrt(2))/14};

            private ChromaSettings m_Settings;
        	private IShader m_ChromaShader;
        	private IFilter m_ChromaScaler;

            protected override string ShaderPath
            {
                get { return "ChromaScaler"; }
            }

            public override ScriptDescriptor Descriptor
            {
                get
                {
                    return new ScriptDescriptor
                    {
                        Guid = new Guid("BDCC94DD-93B3-4414-BA1F-345E10E1C371"),
                        Name = "ChromaScaler",
                        Description = "Chroma Scaler",
                        Copyright = "ChromaScaler by Shiandow",
                        HasConfigDialog = true
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

            public static ChromaScaler Create(Presets preset)
            {
                if (preset == Presets.Custom)
                {
                    throw new ArgumentException(
                        "To use custom preset, call the Create() overload with B and C arguments", "preset");
                }

                return Create((float) BConst[(int) preset], (float) CConst[(int) preset]);
            }

            public static ChromaScaler Create(float B = (float) (1.0/3.0), float C = (float) (1.0/3.0))
            {
                var result = new ChromaScaler();
                result.m_Settings = new ChromaSettings();
                result.m_Settings.Config.B = B;
                result.m_Settings.Config.C = C;
                return result;
            }

            public override void Initialize(int instanceId)
            {
                m_Settings = new ChromaSettings(instanceId);
            }

			public override bool ShowConfigDialog(IWin32Window owner)
            {
                using (var dialog = new ChromaScalerConfigDialog())
                {
                    dialog.Setup(m_Settings.Config);
                    if (dialog.ShowDialog(owner) != DialogResult.OK)
                        return false;

                    OnInputSizeChanged();
                    m_Settings.Save();
                    return true;
                }
            }

            protected override void Dispose(bool disposing)
            {
                DiscardChromaScaler();

                Common.Dispose(ref m_ChromaShader);
            }

            public override void Setup(IRenderer renderer)
            {
                base.Setup(renderer);
				CompileShaders();
                CreateChromaScaler();
            }

            public override IFilter GetFilter()
            {
				return m_ChromaScaler;
            }

            protected override TextureAllocTrigger TextureAllocTrigger
            {
                get { return TextureAllocTrigger.OnInputOutputSizeChanged; }
            }

            private void CompileShaders()
            {
                m_ChromaShader = CompileShader("Chroma.hlsl");
            }

            private void CreateChromaScaler()
            {
            	if (Renderer == null)
            		return;
                
                Func<Size, Size> transformWidth  = s => new Size(Renderer.VideoSize.Width, s.Height);
                Func<Size, Size> transformHeight = s => new Size(s.Width, Renderer.VideoSize.Height);
                Func<Size, Size> transformWidthHeight = s => Renderer.VideoSize;

                var Yinput = new YSourceFilter(Renderer);
                var Uinput = new USourceFilter(Renderer);
                var Vinput = new VSourceFilter(Renderer);

                var chroma = new ChromaFilter(Renderer, m_ChromaShader, m_Settings.Config.B, m_Settings.Config.C, transformWidthHeight, Yinput, Uinput, Vinput);
                var rgb    = new RGBFilter(Renderer, chroma);

                m_ChromaScaler = rgb;
                m_ChromaScaler.Initialize();
            }

            private void DiscardChromaScaler()
            {
                // Disposes the all ancestors too 
                Common.Dispose(ref m_ChromaScaler);
            }
        }

		#region Settings

        public class Settings
        {
            public Settings()
            {
                B = (float) (1.0/3.0);
                C = (float) (1.0/3.0);
                Preset = Presets.Custom;
            }

            [YAXErrorIfMissed(YAXExceptionTypes.Ignore)]
			public float B { get; set; }

            [YAXErrorIfMissed(YAXExceptionTypes.Ignore)]
            public float C { get; set; }

            [YAXErrorIfMissed(YAXExceptionTypes.Ignore)]
            public Presets Preset { get; set; }
        }

        public class ChromaSettings : ScriptSettings<Settings>
        {
            private readonly int m_InstanceId;

            public ChromaSettings(int instanceId)
                : base(false)
            {
                m_InstanceId = instanceId;
                Load();
            }

            public ChromaSettings()
                : base(true)
            {
                Load();
            }

            protected override string ScriptConfigFileName
            {
                get { return string.Format("Shiandow.Chroma.{0}.config", m_InstanceId); }
            }
        }

        #endregion

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
    }
}
