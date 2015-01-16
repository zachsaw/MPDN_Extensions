MPDN Player Extensions
======================

MPDN project page - http://forum.doom9.org/showthread.php?t=171120

<H3>Developers</H3>
<ul>
<li>Zachs</li>
<li>Shiandow</li>
</ul>


How to use Player Extensions?
-----------------------------

To use these player extensions (compatible with ***MPDN v2.18.3*** and above), click the **Download ZIP** button on the right to download the whole repository.

Then extract the files and folders in the PlayerExtensions folder (located in PlayerExtensions-master folder) to your MPDN's PlayerExtensions folder.


Developing / Debugging Player Extensions
----------------------------------------

The easiest way to develop or debug player extensions is to use Microsoft Visual Studio or similar IDEs.

Follow these simple steps:<ol><li>Create a class library</li><li>Add the following assembly references to your project:<ul><li>`Mpdn.Definitions.dll`</li></li></ul></li><li>Set your class library's output folder to MPDN's PlayerExtensions folder</li><li>Set the program to debug your class library to MediaPlayerDotNet.exe</li><li>You're all set! This allows your IDE to run MPDN which in turn loads your class library (PlayerExtension plugin) when you start a debug session</li></ol>

You can set breakpoints and step through your code just as you normally would. Intellisense should work too.