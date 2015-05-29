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

using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using Mpdn.Extensions.RenderScripts.Mpdn.ScriptedRenderChain;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Mpdn.Conditional
    {
        public static class Parser
        {
            private const string FIND_FUNCTION_IDENTIFIERS_PATTERN = @"(\.\s*)?[a-zA-Z_\$]+[a-zA-Z0-9_\$]*\s*(?=(?:[^""\\]*(?:\\.|""(?:[^""\\]*\\.)*[^""\\]*""))*[^""]*$)(?=(?:[^'\\]*(?:\\.|'(?:[^'\\]*\\.)*[^'\\]*'))*[^']*$)";

            public static string BuildCondition(string input)
            {
                var regexCtors = new Regex(FIND_FUNCTION_IDENTIFIERS_PATTERN, RegexOptions.Singleline);

                var result = new StringBuilder();
                var matchCtors = regexCtors.Match(input);
                if (!matchCtors.Success)
                    return input;

                var lastPos = 0;
                do
                {
                    result.Append(input.SubstringIdx(lastPos, matchCtors.Index));
                    var id = matchCtors.Value.Trim();
                    if (!IsKeyword(id))
                    {
                        lastPos = matchCtors.Index;
                    }
                    else
                    {
                        var replacement = "input." + id;
                        result.Append(replacement);
                        lastPos = matchCtors.Index + matchCtors.Value.Length;
                    }

                    matchCtors = matchCtors.NextMatch();

                } while (matchCtors.Success);

                result.Append(input.SubstringIdx(lastPos, input.Length));

                return result.ToString();
            }

            private static bool IsKeyword(string word)
            {
                var propInfos = typeof (IClip).GetProperties();
                return propInfos.Any(pi => pi.Name == word);
            }
        }
    }
}
