MPDN RenderScripts
==================

MediaPlayerDotNet (MPDN) Open Source RenderScripts

MPDN project page - http://forum.doom9.org/showthread.php?t=171120

<H3>Developers</H3>
<ul>
<li>Zachs</li>
<li>Shiandow</li>
</ul>


How to use Render Scripts?
--------------------------

To use these render scripts (compatible with ***MPDN v2.12.0*** and above), click the **Download ZIP** button on the right to download the whole repository.

Then extract the files and folders in the RenderScripts folder (located in RenderScripts-master folder) to your MPDN's RenderScripts folder.


Developing / Debugging Render Scripts
-------------------------------------

The easiest way to develop or debug render scripts is to use Microsoft Visual Studio or similar IDEs.

Follow these simple steps:<ol><li>Create a class library</li><li>Add the following assembly references to your project:<ul><li>`Framework.dll` (from GitHub)</li><li>`RenderScript.dll`<li>`SharpDX.dll`</li><li>`SharpDX.Direct3D9.dll`</li><li>`YAXLib.dll`</li></li></ul></li><li>Set your class library's output folder to MPDN's RenderScripts folder</li><li>Set the program to debug your class library to MediaPlayerDotNet.exe</li><li>You're all set! This allows your IDE to run MPDN which in turn loads your class library (RenderScript plugin) when you start a debug session</li></ol>

You can set breakpoints and step through your code just as you normally would. Intellisense should work too.