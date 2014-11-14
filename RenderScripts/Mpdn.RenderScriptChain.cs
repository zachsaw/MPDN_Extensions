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
            private RenderScript[] m_Scripts = new RenderScript[0];
            private RenderScript[] m_ScriptChain = new RenderScript[0];
            private IFilter m_Filter;

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
                var result = GetScriptChain();
                EnsureScriptsValid(result, "GetScriptChain()");

                return result;
            }

            private void SetupScriptChain()
            {
                var scriptChain = GetScriptChainInternal();
                if (!scriptChain.SequenceEqual(m_ScriptChain))
                {
                    Common.Dispose(ref m_Filter);
                    m_Filter = scriptChain[0].GetFilter();
                    for (int i = 1; i < scriptChain.Count(); i++)
                    {
                        m_Filter = m_Filter.Append(scriptChain[i].GetFilter());
                    }
                    m_Filter.Initialize();
                }
                m_ScriptChain = scriptChain;
                m_Filter.AllocateTextures();
            }

            protected override void Dispose(bool disposing)
            {
                base.Dispose(disposing);

                for (int i = 0; i < m_Scripts.Length; i++)
                {
                    Common.Dispose(ref m_Scripts[i]);
                }
            }

            protected abstract RenderScript[] CreateScripts();

            protected abstract RenderScript[] GetScriptChain();

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

            public override IFilter GetFilter()
            {
                return m_Filter;
            }

            protected override TextureAllocTrigger TextureAllocTrigger
            {
                get { return TextureAllocTrigger.None; }
            }
        }
    }
}
