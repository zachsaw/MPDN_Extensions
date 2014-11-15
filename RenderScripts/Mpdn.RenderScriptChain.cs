using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace Mpdn.RenderScript
{
    namespace Mpdn
    {
        public abstract class RenderScriptChain : RenderScript
        {
            private IList<RenderScript> m_Scripts = new RenderScript[0];
            private IList<RenderScript> m_ScriptChain = new RenderScript[0];

            public override void Setup(IRenderer renderer)
            {
                base.Setup(renderer);

                var scripts = CreateScripts();
                EnsureScriptsValid(scripts, "CreateScripts()");

                m_Scripts = scripts;

                foreach (var s in m_Scripts)
                {
                    s.Setup(renderer);
                }
            }

            private static void EnsureScriptsValid(IEnumerable<RenderScript> scripts, string methodName)
            {
                if (scripts == null || !scripts.Any())
                {
                    throw new InvalidOperationException(methodName +
                                                        " causes an invalid operation by returning null or zero scripts");
                }
            }

            public override void OnInputSizeChanged()
            {
                SetupScriptChain();
            }

            public override void OnOutputSizeChanged()
            {
                SetupScriptChain();
            }

            private RenderScript[] GetScriptChainInternal()
            {
                var result = GetScriptChain().ToArray();
                EnsureScriptsValid(result, "GetScriptChain()");

                return result;
            }

            public override IFilter CreateFilter()
            {
                // RenderScriptChain can't create a filter dynamically (yet)
                return null;
            }

            private void SetupScriptChain()
            {
                var scriptChain = GetScriptChainInternal();
                if (!scriptChain.SequenceEqual(m_ScriptChain))
                {
                    Filter = ChainScripts(scriptChain);
                }
                m_ScriptChain = scriptChain;
                Filter.AllocateTextures();
            }

            private IFilter ChainScripts(IList<RenderScript> scriptChain)
            {
                IFilter previous = SourceFilter;
                for (int i = 1; i < scriptChain.Count(); i++)
                {
                    var script = scriptChain[i];
                    script.SourceFilter.ReplaceWith(previous);
                    previous = script.GetFilter();
                }
                return previous;
            }

            protected override void Dispose(bool disposing)
            {
                base.Dispose(disposing);

                foreach (var script in m_Scripts)
                {
                    Common.Dispose(script);
                }
                m_Scripts = new RenderScript[0];
            }

            protected abstract IList<RenderScript> CreateScripts();

            protected abstract IList<RenderScript> GetScriptChain();

            protected bool IsDownscalingFrom(Size size)
            {
                return !IsNotScalingFrom(size) && !IsUpscalingFrom(size);
            }

            protected bool IsNotScalingFrom(Size size)
            {
                return size == Renderer.TargetSize;
            }

            protected bool IsUpscalingFrom(Size size)
            {
                var targetSize = Renderer.TargetSize;
                return targetSize.Width > size.Width || targetSize.Height > size.Height;
            }

            protected bool IsDownscalingFrom(IList<RenderScript> scriptChain)
            {
                using (var filter = ChainScripts(scriptChain))
                {
                    var size = filter.OutputSize;
                    return IsDownscalingFrom(size);
                }
            }

            protected bool IsUpscalingFrom(IList<RenderScript> scriptChain)
            {
                using (var filter = ChainScripts(scriptChain))
                {
                    var size = filter.OutputSize;
                    return IsUpscalingFrom(size);
                }
            }

            protected bool IsNotScalingFrom(IList<RenderScript> scriptChain)
            {
                using (var filter = ChainScripts(scriptChain))
                {
                    var size = filter.OutputSize;
                    return IsNotScalingFrom(size);
                }
            }

            protected bool IsDownscaling
            {
                get { return !IsNotScaling && !IsUpscaling; }
            }

            protected bool IsNotScaling
            {
                get { return Renderer.VideoSize == Renderer.TargetSize; }
            }

            protected bool IsUpscaling
            {
                get
                {
                    var videoSize = Renderer.VideoSize;
                    var targetSize = Renderer.TargetSize;
                    return targetSize.Width > videoSize.Width || targetSize.Height > videoSize.Height;
                }
            }

            protected override TextureAllocTrigger TextureAllocTrigger
            {
                get { return TextureAllocTrigger.None; }
            }
        }
    }
}
