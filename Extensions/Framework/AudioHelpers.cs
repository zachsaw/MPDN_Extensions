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
// 

using Cudafy;

namespace Mpdn.Extensions.Framework
{
    public static class Decibels
    {
        // 20 / ln( 10 )
        private const float LOG_2_DB = 8.6858896380650365530225783783321f;

        // ln( 10 ) / 20
        private const float DB_2_LOG = 0.11512925464970228420089957273422f;

        /// <summary>
        /// linear to dB conversion
        /// </summary>
        /// <param name="lin">linear value</param>
        /// <returns>decibel value</returns>
        [Cudafy]
        public static float FromLinear(float lin)
        {
            return GMath.Log(lin)*LOG_2_DB;
        }

        /// <summary>
        /// dB to linear conversion
        /// </summary>
        /// <param name="dB">decibel value</param>
        /// <returns>linear value</returns>
        [Cudafy]
        public static float ToLinear(float dB)
        {
            return GMath.Exp(dB*DB_2_LOG);
        }
    }
}