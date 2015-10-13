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

using System;
using Mpdn.Extensions.CustomLinearScalers.Functions;

namespace Mpdn.Extensions.CustomLinearScalers
{
    public class SincJinc : Sinc
    {
        public override Guid Guid
        {
            get { return new Guid("C7D677F7-C2FE-407F-8C7F-CD67A8F3A977"); }
        }

        public override string WindowName
        {
            get { return "Jinc"; }
        }

        public override double GetWindowWeight(double x, double radius)
        {
            return Jinc.CalculateWindow(x, (int) Math.Round(radius));
        }
    }
}
