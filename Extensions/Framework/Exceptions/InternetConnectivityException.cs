using System;

namespace Mpdn.Extensions.Framework.Exceptions
{
    public class InternetConnectivityException : Exception
    {
        public InternetConnectivityException()
        {
        }

        public InternetConnectivityException(string message)
            : base(message)
        {
        }

        public InternetConnectivityException(string message, Exception innerException)
            : base(message, innerException)
        {
        }
    }
}