using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Mpdn.ScriptedRenderChain
    {
        public class ScriptParser
        {
            private const string FIND_COMMENTS_PATTERN = @"(?:\/\*(?:[\s\S]*?)\*\/)|(?:^\s*\/\/(?:.*)$)";
            private const string FIND_FUNCTION_IDENTIFIERS_PATTERN = @"(\.\s*)?[a-zA-Z_\$]+[a-zA-Z0-9_\$]*\s*\((?=(?:[^""\\]*(?:\\.|""(?:[^""\\]*\\.)*[^""\\]*""))*[^""]*$)(?=(?:[^'\\]*(?:\\.|'(?:[^'\\]*\\.)*[^'\\]*'))*[^']*$)";
            private const string FIND_END_MARKERS_PATTERN = @"\)(?=(?:[^""\\]*(?:\\.|""(?:[^""\\]*\\.)*[^""\\]*""))*[^""]*$)(?=(?:[^'\\]*(?:\\.|'(?:[^'\\]*\\.)*[^'\\]*'))*[^']*$)\s*(\;|(\r\n|\r|\n))";
            private const string FIND_PROPERTY_NAMES_PATTERN = @",\s*[a-zA-Z_\$]+[a-zA-Z0-9_\$]\s*\=(?=(?:[^""\\]*(?:\\.|""(?:[^""\\]*\\.)*[^""\\]*""))*[^""]*$)(?=(?:[^'\\]*(?:\\.|'(?:[^'\\]*\\.)*[^'\\]*'))*[^']*$)";

            private readonly HashSet<string> m_FilterTypeNames;

            public ScriptParser(HashSet<string> filterTypeNames)
            {
                m_FilterTypeNames = filterTypeNames;
            }

            public string BuildScript(string contents)
            {
                contents = RemoveComments(contents + Environment.NewLine);
                contents = ReplaceConstructorDecls(contents);
                return contents;
            }

            private static string RemoveComments(string input)
            {
                var regex = new Regex(FIND_COMMENTS_PATTERN, RegexOptions.Multiline);
                return regex.Replace(input, string.Empty);
            }

            private string ReplaceConstructorDecls(string input)
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
                    var ctorId = matchCtors.Value.TrimEnd('(').Trim();
                    if (!m_FilterTypeNames.Contains(ctorId))
                    {
                        if (ctorId == "Import")
                        {
                            // Import is currently a reserved word
                            // Reminder: must prevent check for cyclic imports
                            throw new NotImplementedException();
                        }
                        lastPos = matchCtors.Index;
                    }
                    else
                    {
                        var argsBegin = matchCtors.Index + matchCtors.Value.Length;
                        lastPos = ProcessConstructorArgs(input, argsBegin, ctorId, result);
                    }

                    matchCtors = matchCtors.NextMatch();

                } while (matchCtors.Success);

                result.Append(input.SubstringIdx(lastPos, input.Length));

                return result.ToString();
            }

            private static int ProcessConstructorArgs(string input, int offset, string ctorId, StringBuilder result)
            {
                // find end marker
                var regexEndMarker = new Regex(FIND_END_MARKERS_PATTERN, RegexOptions.Singleline);

                var matchEnds = regexEndMarker.Match(input, offset);
                var args = input.SubstringIdx(offset, matchEnds.Index).TrimEnd(')').Trim();
                args = args.Insert(0, ",");

                // extract property names
                var regexPropNames = new Regex(FIND_PROPERTY_NAMES_PATTERN, RegexOptions.Singleline);

                var matchPropNames = regexPropNames.Matches(args);
                var propNames =
                    (from Match matchPropName in matchPropNames select matchPropName.Value.TrimStart(',').Trim())
                        .ToArray();

                if (!propNames.Any())
                {
                    // No arguments provided
                    result.AppendLine(string.Format("input.Add(new {0}());", ctorId));
                }
                else
                {
                    result.Append("{ ");
                    result.Append(string.Format("__$filter = new {0}(); ", ctorId));
                    for (int i = 0; i < propNames.Length; i++)
                    {
                        var start = matchPropNames[i].Index + matchPropNames[i].Length;
                        var propVal = (i < propNames.Length - 1)
                            ? args.SubstringIdx(start, matchPropNames[i + 1].Index)
                            : args.SubstringIdx(start, args.Length);
                        result.Append(string.Format("__$xhost.AssignProp(__$filter, \"{0}\", {1}); ",
                            propNames[i].TrimEnd('=').Trim(), propVal.Trim()));
                    }
                    result.Append("input.Add(__$filter); }");
                }

                return matchEnds.Index + matchEnds.Length;
            }
        }
    }
}