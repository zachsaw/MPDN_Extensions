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

The easiest way to develop or debug extensions is to use Microsoft Visual Studio or similar IDEs.

Follow these simple steps:

1. Create a class library
1. Add all the DLLs in MPDN's root folder as assembly references to your project
1. Copy the resource files in the Sources folder to Extensions folder
    * e.g. PlaylistForm.resx should be placed in the same folder as PlaylistForm.cs
1. Add all the files from the Extensions folder of this repository to your project
1. Set your class library's output folder to MPDN's Extensions folder
1. Set the program to debug your class library to MediaPlayerDotNet.exe
1. You're all set! This allows your IDE to run MPDN which in turn loads your class library (Extension plugin) when you start a debug session

You can set breakpoints and step through your code just as you normally would. Intellisense should work too.