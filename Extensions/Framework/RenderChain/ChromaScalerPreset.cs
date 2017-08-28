using Mpdn.Extensions.Framework.Chain;
using Mpdn.Extensions.Framework.Config;
using Mpdn.RenderScript;

namespace Mpdn.Extensions.Framework.RenderChain
{
    public class ChromaScalerPreset : Preset<ITextureFilter, IRenderScript>, IChromaScaler
    {
        public ITextureFilter ScaleChroma(ICompositionFilter composition)
        {
            return composition + Chain;
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