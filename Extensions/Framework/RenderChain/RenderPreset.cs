using Mpdn.Extensions.Framework.Chain;
using Mpdn.Extensions.Framework.Config;
using Mpdn.RenderScript;
using SharpDX;

namespace Mpdn.Extensions.Framework.RenderChain
{
    public class ChromaScalerPreset : Preset<ITextureFilter, IRenderScript>, IChromaScaler
    {
        public ITextureFilter CreateChromaFilter(ITextureFilter lumaInput, ITextureFilter chromaInput, TextureSize targetSize, Vector2 chromaOffset)
        {
            var chroma = new ChromaFilter(lumaInput, chromaInput, targetSize, chromaOffset);

            return chroma + Chain;
        }
    }

    public static class ChromaScalerPresetHelper
    {
        public static ChromaScalerPreset MakeNewChromaScalerPreset(this IChainUi<ITextureFilter, IRenderScript> renderScript, string name = null)
        {
            return renderScript.CreateNew().ToChromaScalerPreset();
        }

        public static ChromaScalerPreset ToChromaScalerPreset(this IChainUi<ITextureFilter, IRenderScript> renderScript, string name = null)
        {
            return new ChromaScalerPreset { Name = name ?? renderScript.Descriptor.Name, Script = renderScript };
        }
    }
}