MPDN Extensions
===============

MPDN project page - http://www.zachsaw.com/mpdn/  
MPDN support page on Doom9.org - http://forum.doom9.org/showthread.php?t=171120  
MPDN Extensions Wiki - https://github.com/zachsaw/MPDN_Extensions/wiki

#### Developers
* Zachs
* Shiandow
* DeadlyEmbrace
* Mercy07
* Garteal
* Belphemur


Prerequisites
-------------
* **MPDN v2.46.0** and above
* MPDN Extensions API Level 9
* .NET Framework version 4.0


How to use the Extensions?
--------------------------

To use these extensions, click the **Download ZIP** button on the right sidebar to download the whole repository.

Then extract the files and folders in the Extensions folder into your MPDN Extensions folder (which should be empty when you first installed MPDN - if not, make sure you clean it out first).

If you also want to try out the example scripts, extract the two folders under Examples into your MPDN Extensions folder too.


Developing / Debugging Extensions
---------------------------------

The easiest way to develop or debug extensions is to use Microsoft Visual Studio 2013 or later.

Follow these simple steps:

1. Copy any compatible version of MPDN (AnyCPU edition) into the `MPDN` folder.
    * The VS solution runs `MPDN\MediaPlayerDotNet.exe` when you start a debug session.
1. Open `Mpdn.Extensions.sln`
1. Rebuild the solution
    * Make sure you do this before opening any of the files in the IDE
1. Hit F5 to run MPDN which will load the extensions for debugging

You can set breakpoints and step through your code just as you normally would. Intellisense should work too.

#### Quick Note about .resx Files
Avoid deploying managed resource files (.resx) when possible. If you must deploy them, make sure they go into the `Resources` folder and their filenames match the controls that reference them. For example, `PlaylistForm.resx` is renamed to `Resources\Mpdn.Extensions.PlayerExtensions.Playlist.PlaylistForm.resx` for deployment. The original .resx file must then also be copied into the Source folder and checked in.

Make sure you copy your .resx files into the `Source` folder before running Make.bat (it erases all .resx files in the `Extensions\PlayerExtensions` and `Extensions\RenderScripts` folders).


Releasing Extensions
--------------------

MPDN Extensions are used either as source or compiled.  

As such, it is important that you validate your changes against both the usage models before releasing them on GitHub.

#### Validating Extensions (Compiled Assembly)
Build the solution in the IDE and test your extensions. Also run `make.bat` to make sure the build succeeds outside of the IDE.

#### Validating Extensions (Source Code Deployment)
`Validate Release.bat` sets up MPDN with your extensions to simulate production condition.

