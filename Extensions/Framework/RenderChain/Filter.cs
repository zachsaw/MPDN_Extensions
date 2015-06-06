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
using System.Collections.Generic;
using System.Linq;
using Mpdn.RenderScript;
using TransformFunc = System.Func<Mpdn.Extensions.Framework.RenderChain.TextureSize, Mpdn.Extensions.Framework.RenderChain.TextureSize>;
using IBaseFilter = Mpdn.Extensions.Framework.RenderChain.IFilter<Mpdn.IBaseTexture>;

namespace Mpdn.Extensions.Framework.RenderChain
{
    public interface IFilter<out TTexture>
        where TTexture : class, IBaseTexture
    {
        TTexture OutputTexture { get; }
        TextureSize OutputSize { get; }
        TextureFormat OutputFormat { get; }
        int LastDependentIndex { get; }
        void Render();
        void Reset();
        void Initialize(int time = 1);
        IFilter<TTexture> Compile();
    }

    public interface IFilter : IFilter<ITexture2D>
    {
    }

    public interface IResizeableFilter : IFilter
    {
        void SetSize(TextureSize outputSize);
    }

    public abstract class Filter : IFilter
    {
        protected Filter(params IBaseFilter[] inputFilters)
        {
            if (inputFilters == null || inputFilters.Any(f => f == null))
            {
                throw new ArgumentNullException("inputFilters");
            }

            Initialized = false;
            CompilationResult = null;
            InputFilters = inputFilters;
        }

        protected abstract void Render(IList<IBaseTexture> inputs);

        #region IFilter Implementation

        private bool Updated { get; set; }
        private bool Initialized { get; set; }
        private int FilterIndex { get; set; }

        protected IFilter<ITexture2D> CompilationResult { get; set; }
        protected ITargetTexture OutputTarget { get; private set; }

        public IBaseFilter[] InputFilters { get; private set; }

        public ITexture2D OutputTexture { get { return OutputTarget; } }
        public abstract TextureSize OutputSize { get; }
        public virtual TextureFormat OutputFormat
        {
            get { return Renderer.RenderQuality.GetTextureFormat(); }
        }
        
        public int LastDependentIndex { get; private set; }

        public void Initialize(int time = 1)
        {
            LastDependentIndex = time;

            if (Initialized)
                return;

            foreach (var f in InputFilters)
            {
                f.Initialize(LastDependentIndex);
                LastDependentIndex = f.LastDependentIndex;
            }

            FilterIndex = LastDependentIndex;

            foreach (var filter in InputFilters)
            {
                filter.Initialize(FilterIndex);
            }

            LastDependentIndex++;

            Initialized = true;
        }

        public IFilter<ITexture2D> Compile()
        {
            if (CompilationResult != null) 
                return CompilationResult;

            for (int i = 0; i < InputFilters.Length; i++)
            {
                InputFilters[i] = InputFilters[i].Compile();
            }

            CompilationResult = Optimize();

            return CompilationResult;
        }

        protected virtual IFilter<ITexture2D> Optimize()
        {
            return this;
        }

        public void Render()
        {
            if (Updated)
                return;

            Updated = true;

            foreach (var filter in InputFilters)
            {
                filter.Render();
            }

            var inputTextures =
                InputFilters
                    .Select(f => f.OutputTexture)
                    .ToList();

            OutputTarget = TexturePool.GetTexture(OutputSize, OutputFormat);

            Render(inputTextures);

            foreach (var filter in InputFilters)
            {
                if (filter.LastDependentIndex <= FilterIndex)
                {
                    filter.Reset();
                }
            }
        }

        public void Reset()
        {
            Updated = false;

            if (OutputTarget != null)
            {
                TexturePool.PutTexture(OutputTarget);
            }

            OutputTarget = null;
        }

        #endregion
    }
}