using System;
using System.ComponentModel;
using System.Linq;
using System.Windows.Forms;
using YAXLib;

namespace Mpdn.RenderScript
{
    namespace Mpdn.ImageProcessor
    {
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

        public class ImageProcessor : RenderChain
        {
            #region Settings

            private string[] m_ShaderFileNames;

            public ImageProcessor()
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

            #endregion

            protected override string ShaderPath
            {
                get { return "ImageProcessingShaders"; }
            }

            public string FullShaderPath { get { return ShaderDataFilePath; } }

            public override IFilter CreateFilter(IFilter sourceFilter)
            {
                if (UseImageProcessor)
                    return ShaderFileNames.Aggregate(sourceFilter, (current, filename) => CreateFilter(CompileShader(filename), current));
                else
                    return sourceFilter;
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

                    var usage = ImageProcessorUsage;
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
        }

        public class ImageProcessorScript : ConfigurableRenderChainUi<ImageProcessor, ImageProcessorConfigDialog>
        {
            protected override string ConfigFileName
            {
                get { return "Mpdn.ImageProcessor"; }
            }

            protected override RenderScriptDescriptor ScriptDescriptor
            {
                get
                {
                    return new RenderScriptDescriptor
                    {
                        Guid = new Guid("50CA262F-65B6-4A0F-A8B5-5E25B6A18217"),
                        Name = "Image Processor",
                        Description = GetDescription(),
                    };
                }
            }

            private string GetDescription()
            {
                return ScriptConfig == null || Chain.ShaderFileNames.Length == 0
                    ? "Pixel shader pre-/post-processing filter"
                    : GetUsageString() + string.Join(" ➔ ", Chain.ShaderFileNames);
            }

            private string GetUsageString()
            {
                var usage = Chain.ImageProcessorUsage;
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
        }
    }
}
