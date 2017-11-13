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
using System.Diagnostics;
using Mpdn.Extensions.Framework.Filter;

namespace Mpdn.Extensions.Framework.Chain
{
    public abstract class FilterChainScript<TFilter, TValue> : IScript, IDisposable
        where TFilter : IFilterBase<IFilterOutput<TValue>>
    {
        protected abstract void OutputResult(TValue result);

        protected abstract TFilter MakeInitialFilter();

        protected virtual TFilter FinalizeOutput(TFilter output) { return output; }

        protected abstract TFilter HandleError(Exception e);

        #region Implementation

        private TFilter m_SourceFilter;
        private TFilter m_Filter;

        private readonly Chain<TFilter> m_Chain;

        protected FilterChainScript(Chain<TFilter> chain)
        {
            m_Chain = chain;
            Status = string.Empty;
        }

        public string Status { get; private set; }

        public void Update()
        {
            UpdateFilter();

            UpdateStatus();
        }

        public void UpdateFilter()
        {
            var oldFilter = m_Filter;
            DisposeHelper.Dispose(ref m_SourceFilter);

            try
            {
                var input = MakeInitialFilter();
                m_Filter = FinalizeOutput(m_Chain.Process(input));
            }
            catch (Exception ex)
            {
                m_Filter = HandleError(ex);
                m_Filter.AddLabel(ErrorMessage(ex));
            }
            finally
            {
                DisposeHelper.Dispose(ref oldFilter);
            }
        }

        private void UpdateStatus()
        {
            Status = "Status Invalid";
            if (m_Filter != null)
                Status = m_Filter.ProcessData.CreateString();
        }

        public virtual bool Execute()
        {
            try
            {
                m_Filter.Extract(OutputResult);

                return true;
            }
            catch (Exception e)
            {
                var message = ErrorMessage(e);
                Trace.WriteLine(message);
                Status = message;
                return false;
            }
        }

        #endregion

        #region Error Handling

        protected static Exception InnerMostException(Exception e)
        {
            while (e.InnerException != null)
            {
                e = e.InnerException;
            }

            return e;
        }

        protected string ErrorMessage(Exception e)
        {
            var ex = InnerMostException(e);
            return string.Format("Error in {0}:\r\n\r\n{1}\r\n\r\n~\r\nStack Trace:\r\n{2}",
                    GetType().Name, ex.Message, ex.StackTrace);
        }

        #endregion

        #region Resource Management

        ~FilterChainScript()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            DisposeHelper.Dispose(m_Filter);
            DisposeHelper.Dispose(ref m_SourceFilter);
        }

        #endregion
    }
}
