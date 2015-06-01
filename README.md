MPDN Extensions
===============

MPDN project page - http://forum.doom9.org/showthread.php?t=171120

<H3>Developers</H3>
* Zachs
* Shiandow
* DeadlyEmbrace
* Mercy07
* Garteal
* Belphemur


Prerequisites
-------------
* **MPDN v2.26.0** and above
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

1. Copy any compatible version of MPDN (any edition is fine) into the `Sources\Solution\MPDN` folder.
    * The VS solution runs `Sources\Solution\MPDN\MediaPlayerDotNet.exe` when you start a debug session.
1. Open Mpdn.Extensions.sln
1. Rebuild Mpdn.Extensions
    * Make sure you do this before opening any of the files in the IDE
1. Hit F5 to run MPDN which will load the extensions for debugging

You can set breakpoints and step through your code just as you normally would. Intellisense should work too.