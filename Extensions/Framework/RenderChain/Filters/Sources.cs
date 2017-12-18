// This file is a part of MPDN Extensions.
// https://github.com/zachsaw/MPDN_Extensions
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
// 
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public
// License along with this library.

using System;
using SharpDX;
using Shiandow.Lending;
using Mpdn.Extensions.Framework.Filter;
using Mpdn.RenderScript;
using Size = System.Drawing.Size;

namespace Mpdn.Extensions.Framework.RenderChain.Filters
{
    using static FilterHelper;
    using static FilterBaseHelper;
    using static FilterOutputHelper;

    public class JustTextureOutput<TTexture> : ITextureOutput<TTexture>, ITextureDescription, IDisposable
        where TTexture : IBaseTexture
    {
        public JustTextureOutput(TTexture texture)
        {
            if (texture == null)
                throw new ArgumentNullException("texture");

            m_Texture = texture;
        }

        private readonly TTexture m_Texture;
        public virtual TextureSize Size { get { return m_Texture.GetSize(); } }
        public virtual TextureFormat Format { get { return m_Texture.Format; } }

        public ITextureDescription Description { get { return this; } }

        ILease<TTexture> ILendable<TTexture>.GetLease()
        {
            return LeaseHelper.Return(m_Texture);
        }

        public void Dispose()
        {
            m_Texture.Dispose();
        }
    }

    public class DeferredTextureOutput<TTexture> : ITextureOutput<TTexture>, ITextureDescription
        where TTexture : IBaseTexture
    {
        private readonly Func<TextureSize> m_Size;
        private readonly Func<TextureFormat> m_Format;
        private readonly Func<TTexture> m_TextureFunc;

        public DeferredTextureOutput(Func<TTexture> textureFunc, TextureSize size)
            : this(textureFunc, () => size)
        { }

        public DeferredTextureOutput(Func<TTexture> textureFunc, TextureSize size, TextureFormat format)
            : this(textureFunc, () => size, () => format)
        { }

        public DeferredTextureOutput(Func<TTexture> textureFunc, Func<TextureSize> size = null, Func<TextureFormat> format = null)
        {
            m_TextureFunc = textureFunc;
            m_Size = size ?? (() => new TextureSize(0,0));
            m_Format = format ?? (() => Renderer.RenderQuality.GetTextureFormat());
        }

        public TTexture Texture
        {
            get
            {
                return m_TextureFunc();
            }
        }

        public ITextureDescription Description { get { return this; } }

        public TextureSize Size
        {
            get { return Texture == null ? m_Size() : Texture.GetSize(); }
        }

        public TextureFormat Format
        {
            get { return Texture == null ? m_Format() : Texture.Format; }
        }

        ILease<TTexture> ILendable<TTexture>.GetLease()
        {
            return LeaseHelper.Return(Texture);
        }
    }
    
    public class TextureSourceFilter<TTexture> : Filter<ITextureDescription, TTexture>, ITextureFilter<TTexture>
        where TTexture : IBaseTexture
    {
        public TextureSourceFilter(TTexture texture)
            : this(new JustTextureOutput<TTexture>(texture))
        { }

        public TextureSourceFilter(ITextureOutput<TTexture> value)
            : base(Return(value))
        { }
    }

    public sealed class NullFilter : TextureSourceFilter<ITexture2D>, ITextureFilter
    {
        public NullFilter() : base(new DeferredTextureOutput<ITexture2D>(() => Renderer.OutputRenderTarget, Renderer.TargetSize)) { }
    }

    public sealed class YSourceFilter : TextureSourceFilter<ITexture2D>, ITextureFilter
    {
        public YSourceFilter() : base(new DeferredTextureOutput<ITexture2D>(() => Renderer.TextureY, Renderer.LumaSize)) { }
    }

    public sealed class USourceFilter : TextureSourceFilter<ITexture2D>, ITextureFilter
    {
        public USourceFilter() : base(new DeferredTextureOutput<ITexture2D>(() => Renderer.TextureU, Renderer.ChromaSize)) { }
    }

    public sealed class VSourceFilter : TextureSourceFilter<ITexture2D>, ITextureFilter
    {
        public VSourceFilter() : base(new DeferredTextureOutput<ITexture2D>(() => Renderer.TextureV, Renderer.ChromaSize)) { }
    }

    public interface ISourceFilter : IResizeableFilter, ICanUndo<RgbProcess> { }

    public interface ISouceCompositionFilter : ISourceFilter, ICompositionFilter
    {
        ISourceFilter FixComposition();
    }

    public sealed class VideoSourceFilter : TextureFilter, ISouceCompositionFilter
    {
        private readonly bool m_WantYuv;
        private readonly TrueSourceFilter m_TrueSource;

        public VideoSourceFilter(IRenderScript script) : this(new TrueSourceFilter(script)) { }

        #region ISourceFilter Implementation

        public void EnableTag() { }

        ITextureFilter ICanUndo<RgbProcess>.Undo()
        {
            return new VideoSourceFilter(m_TrueSource, Output.Size, true);
        }

        public ITextureFilter ResizeTo(TextureSize outputSize)
        {
            return new VideoSourceFilter(m_TrueSource, outputSize, m_WantYuv);
        }
        
        #endregion

        #region Labeling

        public static string ChromaScaleDescription(TextureSize size)
        {
            var chromaConvolver = Renderer.ChromaOffset.IsZero ? null : Renderer.ChromaUpscaler;
            var chromastatus = StatusHelpers.ScaleDescription(Renderer.ChromaSize, size, Renderer.ChromaUpscaler, Renderer.ChromaDownscaler, chromaConvolver);

            return chromastatus.AddPrefixToDescription("Chroma: ");
        }

        public static string LumaScaleDescription(TextureSize size)
        {
            var lumastatus = StatusHelpers.ScaleDescription(Renderer.VideoSize, size, Renderer.LumaUpscaler, Renderer.LumaDownscaler);

            return lumastatus.AddPrefixToDescription("Luma: ");
        }

        #endregion

        #region Composition Handling

        private ICompositionFilter SourceComposition { get { return m_TrueSource.Composition; } }

        // Note: accesing any of these disables internal scalers
        public ITextureFilter Luma { get { return SourceComposition.Luma; } }
        public ITextureFilter Chroma { get { return SourceComposition.Chroma; } }
        public TextureSize TargetSize { get { return SourceComposition.TargetSize; } }
        public Vector2 ChromaOffset { get { return SourceComposition.ChromaOffset; } }

        public ISourceFilter FixComposition()
        {
            return new FixedSource(this);
        }

        private class FixedSource : TextureFilter, ISourceFilter
        {
            public FixedSource(VideoSourceFilter source)
                : base(source)
            {
                m_Source = source;
            }

            #region ISourceFilter Implementation

            private readonly VideoSourceFilter m_Source;

            private ITextureFilter FixComposition(ITextureFilter filter)
            {
                return (filter as ISouceCompositionFilter).FixComposition();
            }

            ITextureFilter ICanUndo<RgbProcess>.Undo()
            {
                return FixComposition(((ICanUndo<RgbProcess>)m_Source).Undo());
            }

            public ITextureFilter ResizeTo(TextureSize outputSize)
            {
                return FixComposition(m_Source.ResizeTo(outputSize));
            }

            public void EnableTag()
            {
                m_Source.EnableTag();
            }

            #endregion
        }

        #endregion

        #region SourceFilter Implementation

        public ScriptInterfaceDescriptor Descriptor
        {
            get { return m_TrueSource.Descriptor; }
        }

        private VideoSourceFilter(TrueSourceFilter trueSource, TextureSize? outputSize = null, bool? wantYuv = null)
            : this(trueSource, outputSize ?? trueSource.Descriptor.PrescaleSize, wantYuv ?? trueSource.IsYuv())
        { }

        private VideoSourceFilter(TrueSourceFilter trueSource, TextureSize outputSize, bool wantYuv)
            : base(from _ in trueSource
                   from result in Compile(new TextureDescription(outputSize), () =>
                   {
                       ITextureFilter source = trueSource;

                       if (trueSource.IsYuv() && !wantYuv)
                           source = source.ConvertToYuv();

                       if (!trueSource.IsYuv() && wantYuv)
                           source = source.ConvertToRgb();

                       return source.SetSize(outputSize);
                   })
                   select result)
        {
            m_TrueSource = trueSource;
            m_WantYuv = wantYuv;

            if (m_WantYuv)
                m_TrueSource.WantYuv = true; // Prefer enabling (generates less overhead)

            m_TrueSource.PrescaleSize = Output.Size; // Try change source size, always use latest value
        }

        #endregion

        #region TrueSourceFilter Class

        private sealed class TrueSourceFilter : TextureFilter
        {
            public bool IsYuv() { return m_RenderScript.Descriptor.WantYuv; }

            public bool WantYuv { private get; set; }
            public TextureSize? PrescaleSize { private get; set; }

            public ICompositionFilter Composition
            {
                get
                {
                    m_Composition = m_Composition ?? (m_Composition = SourceComposition());
                    m_Composition = (ICompositionFilter)m_Composition.SetSize(Descriptor.PrescaleSize);
                    return m_Composition;
                }
            }

            public ScriptInterfaceDescriptor Descriptor
            {
                get
                {
                    return new ScriptInterfaceDescriptor
                    {
                        WantYuv = WantYuv && Renderer.InputFormat.IsYuv(),
                        Prescale = m_Composition == null,
                        PrescaleSize = (Size)(PrescaleSize ?? Renderer.VideoSize)
                    };
                }
            }

            #region Implementation

            private readonly IRenderScript m_RenderScript;
            private ICompositionFilter m_Composition;

            public static CompositionFilter SourceComposition()
            {
                return new CompositionFilter(new YSourceFilter(), new ChromaSourceFilter());
            }

            private class UnsafeTextureDescription : ITextureDescription, IEquatable<ITextureDescription>
            {
                private Func<Size> m_Size;

                public UnsafeTextureDescription(Func<Size> func)
                {
                    m_Size = func;
                }

                public TextureSize Size { get { return m_Size(); } }
                public TextureFormat Format { get { throw new NotImplementedException(); } }

                public bool Equals(ITextureDescription other) { return m_Size() == other.Size; }
            }

            public TrueSourceFilter(IRenderScript script)
                : base(Compile(
                    new UnsafeTextureDescription(() => script.Descriptor.PrescaleSize), 
                    () =>
                    {
                        if (script.Descriptor.Prescale)
                            return new TextureFilter(Return(
                                new DeferredTextureOutput<ITexture2D>(
                                    () => Renderer.InputRenderTarget,
                                    () => script.Descriptor.PrescaleSize)))
                                    .Labeled(ChromaScaleDescription(script.Descriptor.PrescaleSize).AddPostfixToDescription(" (internal)"))
                                    .Labeled(  LumaScaleDescription(script.Descriptor.PrescaleSize).AddPostfixToDescription(" (internal)"));

                        ICanUndo<RgbProcess> source = SourceComposition();
                        return (script.Descriptor.WantYuv ? source.Undo() : source)
                            .SetSize(script.Descriptor.PrescaleSize);
                    }))
            {
                m_RenderScript = script;
                WantYuv = false;
            }

            #endregion
        }

        #endregion
    }
}
