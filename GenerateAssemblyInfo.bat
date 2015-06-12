@echo off
echo using System.Reflection;
echo using System.Runtime.CompilerServices;
echo using System.Runtime.InteropServices;
echo.
echo // General Information about an assembly is controlled through the following 
echo // set of attributes. Change these attribute values to modify the information
echo // associated with an assembly.
echo [assembly: AssemblyTitle("MPDN Extensions")]
echo [assembly: AssemblyDescription("Open Source Extensions for Media Player .NET (MPDN)")]
echo [assembly: AssemblyConfiguration("")]
echo [assembly: AssemblyCompany("MPDN Extensions Team")]
echo [assembly: AssemblyProduct("MPDN Extensions")]
echo [assembly: AssemblyCopyright("LGPL-3.0 (See github.com/zachsaw/MPDN_Extensions)")]
echo [assembly: AssemblyTrademark("")]
echo [assembly: AssemblyCulture("")]
echo. 
echo // Setting ComVisible to false makes the types in this assembly not visible 
echo // to COM components.  If you need to access a type in this assembly from 
echo // COM, set the ComVisible attribute to true on that type.
echo [assembly: ComVisible(false)]
echo. 
echo // The following GUID is for the ID of the typelib if this project is exposed to COM
echo [assembly: Guid("6223ffff-b668-4829-ae95-beb5639c83fb")]
echo. 
echo // Version information for an assembly consists of the following four values:
echo //
echo //      Major Version
echo //      Minor Version 
echo //      Build Number
echo //      Revision
echo //
echo [assembly: AssemblyVersion("0.0.0.0")]
echo [assembly: AssemblyFileVersion("%1")]

:Quit