using System;
using System.ComponentModel;
using System.Linq;
using System.Windows.Forms;
using YAXLib;

namespace Mpdn.RenderScript
{
    namespace Mpdn.ImageProcessor
    {
        #region Settings

        public class Settings
        {
            private string[] m_ShaderFileNames;

            public Settings()
            {
                ImageProcessorUsage = ImageProcessorUsage.Always;
            }

            [YAXErrorIfMissed(YAXExceptionTypes.Ignore)]
            public string[] ShaderFileNames
            {
                get { return m_ShaderFileNames ?? (m_ShaderFileNames = new string[0]); }
                set { m_ShaderFileNames = value; }
            }

            [YAXErrorIfMissed(YAXExceptionTypes.Ignore)]
            public ImageProcessorUsage ImageProcessorUsage { get; set; }

            public string ShaderPath; // Not saved
        }

        #endregion

        #region ImageProcessorUsage

        public enum ImageProcessorUsage
        {
            [Description("Always")]
            Always,
            [Description("Never")]
            Never,
            [Description("When upscaling video")]
            WhenUpscaling,
            [Description("When downscaling video")]
            WhenDownscaling,
            [Description("When not scaling video")]
            WhenNotScaling,
            [Description("When upscaling input")]
            WhenUpscalingInput,
            [Description("When downscaling input")]
            WhenDownscalingInput,
            [Description("When not scaling input")]
            WhenNotScalingInput
        }

        #endregion

        public class ImageProcessor : ConfigurableRenderScript<Settings, ImageProcessorConfigDialog>
        {
            private IFilter m_ImageFilter;
            private IShader[] m_Shaders = new IShader[0];
            private string[] m_ShaderFileNames = new string[0];

            public static ImageProcessor Create(params string[] shaderFileNames)
            {
                var result = new ImageProcessor();
                result.Initialize();
                result.Settings.Config.ShaderFileNames = shaderFileNames;
                return result;
            }

            protected override ConfigurableRenderScriptDescriptor ConfigScriptDescriptor
            {
                get
                {
                    return new ConfigurableRenderScriptDescriptor
                    {
                        Guid = new Guid("50CA262F-65B6-4A0F-A8B5-5E25B6A18217"),
                        Name = "Image Processor",
                        Description = GetDescription(),
                        ConfigFileName = "Mpdn.ImageProcessor"
                    };
                }
            }

            protected override string ShaderPath
            {
                get { return "ImageProcessingShaders"; }
            }

            protected override TextureAllocTrigger TextureAllocTrigger
            {
                get { return TextureAllocTrigger.OnOutputSizeChanged; }
            }

            public override IFilter CreateFilter(Settings settings)
            {
                SetupRenderChain();
                return m_ImageFilter;
            }

            protected override void Initialize(Config settings)
            {
                settings.Config.ShaderPath = ShaderDataFilePath;
            }

            public override bool ShowConfigDialog(IWin32Window owner)
            {
                if (!base.ShowConfigDialog(owner))
                    return false;

                SetupRenderChain();
                OnOutputSizeChanged();
                return true;
            }

            protected override void Dispose(bool disposing)
            {
                Common.Dispose(ref m_ImageFilter);
                DisposeShaders();
            }

            public override IFilter GetFilter()
            {
                lock (Settings)
                {
                    return UseImageProcessor ? m_ImageFilter : SourceFilter;
                }
            }

            private bool UseImageProcessor
            {
                get
                {
                    bool notscalingVideo = false;
                    bool upscalingVideo = false;
                    bool downscalingVideo = false;
                    bool notscalingInput = false;
                    bool upscalingInput = false;
                    bool downscalingInput = false;

                    var usage = Settings.Config.ImageProcessorUsage;
                    var inputSize = Renderer.VideoSize;
                    var outputSize = Renderer.TargetSize;
                    if (outputSize == inputSize)
                    {
                        // Not scaling video
                        notscalingVideo = true;
                    }
                    else if (outputSize.Width > inputSize.Width)
                    {
                        // Upscaling video
                        upscalingVideo = true;
                    }
                    else
                    {
                        // Downscaling video
                        downscalingVideo = true;
                    }
                    inputSize = Renderer.InputSize;
                    outputSize = Renderer.OutputSize;
                    if (outputSize == inputSize)
                    {
                        // Not scaling input
                        notscalingInput = true;
                    }
                    else if (outputSize.Width > inputSize.Width)
                    {
                        // Upscaling input
                        upscalingInput = true;
                    }
                    else
                    {
                        // Downscaling input
                        downscalingInput = true;
                    }

                    switch (usage)
                    {
                        case ImageProcessorUsage.Always:
                            return true;
                        case ImageProcessorUsage.Never:
                            return false;
                        case ImageProcessorUsage.WhenUpscaling:
                            return upscalingVideo;
                        case ImageProcessorUsage.WhenDownscaling:
                            return downscalingVideo;
                        case ImageProcessorUsage.WhenNotScaling:
                            return notscalingVideo;
                        case ImageProcessorUsage.WhenUpscalingInput:
                            return upscalingInput;
                        case ImageProcessorUsage.WhenDownscalingInput:
                            return downscalingInput;
                        case ImageProcessorUsage.WhenNotScalingInput:
                            return notscalingInput;
                        default:
                            throw new ArgumentOutOfRangeException();
                    }
                }
            }

            private string GetDescription()
            {
                return Settings == null || Settings.Config.ShaderFileNames.Length == 0
                    ? "Pixel shader pre-/post-processing filter"
                    : GetUsageString() + string.Join(" ➔ ", Settings.Config.ShaderFileNames);
            }

            private string GetUsageString()
            {
                var usage = Settings.Config.ImageProcessorUsage;
                string result;
                switch (usage)
                {
                    case ImageProcessorUsage.Never:
                        result = "[INACTIVE] ";
                        break;
                    case ImageProcessorUsage.WhenUpscaling:
                        result = "When upscaling video: ";
                        break;
                    case ImageProcessorUsage.WhenDownscaling:
                        result = "When downscaling video: ";
                        break;
                    case ImageProcessorUsage.WhenNotScaling:
                        result = "When not scaling video: ";
                        break;
                    case ImageProcessorUsage.WhenUpscalingInput:
                        result = "When upscaling input: ";
                        break;
                    case ImageProcessorUsage.WhenDownscalingInput:
                        result = "When downscaling input: ";
                        break;
                    case ImageProcessorUsage.WhenNotScalingInput:
                        result = "When not scaling input: ";
                        break;
                    default:
                        result = string.Empty;
                        break;
                }
                return result;
            }

            private void SetupRenderChain()
            {
                lock (Settings)
                {
                    if (Renderer == null)
                        return;

                    var shaderFileNames = Settings.Config.ShaderFileNames;
                    if (!shaderFileNames.SequenceEqual(m_ShaderFileNames))
                    {
                        CompileShaders(shaderFileNames);
                    }

                    Common.Dispose(ref m_ImageFilter);

                    m_ImageFilter = m_Shaders.Aggregate((IFilter) SourceFilter, (current, shader) => CreateFilter(shader, current));
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
    }
}
