MPDN Extensions
===============

MPDN project page - http://forum.doom9.org/showthread.php?t=171120

<H3>Developers</H3>
<ul>
<li>Zachs</li>
<li>Shiandow</li>
<li>DeadlyEmbrace</li>
<li>Mercy07</li>
<li>Garteal</li>
</ul>


How to use Extensions?
----------------------

To use these extensions (compatible with ***MPDN v2.25.4*** and above), click the **Download ZIP** button on the right to download the whole repository.

Then extract the files and folders in the Extensions folder to your MPDN's Extensions folder.


Developing / Debugging Extensions
---------------------------------

The easiest way to develop or debug extensions is to use Microsoft Visual Studio or similar IDEs.

Follow these simple steps:
<ol>
<li>Create a class library</li>
<li>Add all the DLLs in MPDN's root folder as assembly references to your project</li>
<li>Copy the resource files in the Sources folder to Extensions folder (e.g. PlaylistForm.resx should be placed in the same folder as PlaylistForm.cs)</li>
<li>Add all the files from the Extensions folder of this repository to your project</li>
<li>Set your class library's output folder to MPDN's Extensions folder</li>
<li>Set the program to debug your class library to MediaPlayerDotNet.exe</li>
<li>You're all set! This allows your IDE to run MPDN which in turn loads your class library (Extension plugin) when you start a debug session</li>
</ol>

You can set breakpoints and step through your code just as you normally would. Intellisense should work too.