; This file is a part of MPDN Extensions.
; https://github.com/zachsaw/MPDN_Extensions
;
; This library is free software; you can redistribute it and/or
; modify it under the terms of the GNU Lesser General Public
; License as published by the Free Software Foundation; either
; version 3.0 of the License, or (at your option) any later version.
; 
; This library is distributed in the hope that it will be useful,
; but WITHOUT ANY WARRANTY; without even the implied warranty of
; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
; Lesser General Public License for more details.
; 
; You should have received a copy of the GNU Lesser General Public
; License along with this library.
; 

SetCompressor lzma

; Modern user interface
!include "MUI2.nsh"

; Install for all users. MultiUser.nsh also calls SetShellVarContext to point 
; the installer to global directories (e.g. Start menu, desktop, etc.)
!define MULTIUSER_EXECUTIONLEVEL Admin
!include "MultiUser.nsh"

!include "WinMessages.nsh"

!addplugindir ./
!include "nsProcess.nsh"

; x64.nsh for architecture detection
!include "x64.nsh"

; File Associations
!include "FileAssociation.nsh"

; MD5 
!include "md5.nsh"

; Read the command-line parameters
!insertmacro GetParameters
!insertmacro GetOptions

!getdllversion "TEMP\Extensions\Mpdn.Extensions.dll" VERSION_

Var /GLOBAL mpdn32_root
Var /GLOBAL mpdn64_root
Var /GLOBAL mpdn_root
Var /GLOBAL uninstallerPresent
Var /Global doCleanInstall

;--------------------------------
; Advanced Installer Variables
Var /Global playerArchitecture
Var /Global playerInstallerPath
;--------------------------------

;Configuration

;General

; Package name as shown in the installer GUI
Name "${PROJECT_NAME} v${VERSION_1}.${VERSION_2}.${VERSION_3}"

; Installer filename
OutFile "${PROJECT_NAME}_v${VERSION_1}.${VERSION_2}.${VERSION_3}_Installer.exe"

ShowInstDetails show
ShowUninstDetails show

;--------------------------------
;Modern UI Configuration

; Compile-time constants which we'll need during install
!define MUI_WELCOMEPAGE_TEXT "This wizard will guide you through the installation of ${PROJECT_NAME} v${VERSION_1}.${VERSION_2}.${VERSION_3}."

!define MUI_COMPONENTSPAGE_TEXT_TOP "Select the components to install/upgrade.  Stop any MPDN processes.$\r$\n*** WARNING ***  Existing extensions (if any) will be removed and replaced."

!define MUI_COMPONENTSPAGE_SMALLDESC

!define MUI_FINISHPAGE_NOAUTOCLOSE
!define MUI_ABORTWARNING
!define MUI_HEADERIMAGE

!define MUI_PAGE_CUSTOMFUNCTION_SHOW Welcome.show
!define MUI_PAGE_CUSTOMFUNCTION_LEAVE Welcome.leave
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "../LGPL-3.0.txt"
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

;--------------------------------
;Languages
 
!insertmacro MUI_LANGUAGE "English"
  
;--------------------------------
;Language Strings

LangString DESC_SecMPDNExtensions32 ${LANG_ENGLISH} "Install ${PROJECT_NAME} for MPDN 32-bit Edition."
LangString DESC_SecMPDNExtensions64 ${LANG_ENGLISH} "Install ${PROJECT_NAME} for MPDN 64-bit Edition."

;--------------------------------
;Macros

!macro InstallExtensions path
    StrCpy $uninstallerPresent "0"
    IfFileExists "${path}\Extensions\Uninstall.exe" 0 noUn
        StrCpy $uninstallerPresent "1"
    noUn:    
    ${If} $doCleanInstall == "1"
        RMDir /r "${path}\Extensions"
    ${Else}
        ${IfNot} $uninstallerPresent == "1"
            RMDir /r "${path}\InstTemp"
            Rename "${path}\Extensions\RenderScripts\ImageProcessingShaders" "${path}\InstTemp"
            RMDir /r "${path}\Extensions"
            SetOverwrite on
        ${Else}
            ExecWait "${path}\Extensions\Uninstall.exe /S _?=${path}\Extensions"
            Delete "${path}\Extensions\Uninstall.exe"
            SetOverwrite off
        ${EndIf}
    ${EndIf}    
    SetOutPath "${path}"
    File /r "TEMP\*.*"
    ${IfNot} $uninstallerPresent == "1"
         ${IfNot} $doCleanInstall == "1"
            IfFileExists "${path}\InstTemp\*.*" 0 skipRestore
            RMDir /r "${path}\Extensions\RenderScripts\ImageProcessingShaders"
            Rename "${path}\InstTemp" "${path}\Extensions\RenderScripts\ImageProcessingShaders"
            CreateDirectory "${path}\Extensions\RenderScripts\ImageProcessingShaders"
        skipRestore:
        ${EndIf}
    ${EndIf}
    WriteUninstaller "${path}\Extensions\Uninstall.exe"
!macroend

;--------------------
;Pre-install section

Section -pre
    ${nsProcess::FindProcess} "MediaPlayerDotNet.exe" $R0
    ${If} $R0 == 0
        MessageBox MB_YESNO|MB_ICONEXCLAMATION "To perform the specified operation, MPDN needs to be closed.$\r$\n$\r$\nClose it now?" /SD IDYES IDNO guiEndNo
        DetailPrint "Closing MPDN..."
        Goto guiEndYes
    ${Else}
        Goto mpdnNotRunning
    ${EndIf}

    guiEndNo:
        Quit

    guiEndYes:
        ; user wants to close MPDN as part of install/upgrade
        ${nsProcess::FindProcess} "MediaPlayerDotNet.exe" $R0
        ${If} $R0 == 0
            ${nsProcess::KillProcess} "MediaPlayerDotNet.exe" $R0
        ${Else}
            Goto guiClosed
        ${EndIf}
        Sleep 100
        Goto guiEndYes

    guiClosed:
    
    mpdnNotRunning:    

SectionEnd

Section /o "-Install Player" SecInstallPlayer
    DetailPrint "Install Player"
    IfFileExists $playerInstallerPath 0 noPlayer
        Banner::show /set 76 "Installing the Player" "MediaPlayerDotNet"

        Banner::getWindow
        Pop $1
        
        ExecWait "$playerInstallerPath /S"
        ;Delete $playerInstallerPath
        
        Banner::destroy        
        GoTo playedInstalled
    noPlayer:  
        DetailPrint "Player installer not found: $playerInstallerPath"
        
    playedInstalled:
        
SectionEnd

Section /o "Extensions for MPDN x86" SecMPDNExtensions32
    !insertmacro InstallExtensions "$mpdn32_root"

SectionEnd

Section /o "Extensions for MPDN x64" SecMPDNExtensions64
    !insertmacro InstallExtensions "$mpdn64_root"
SectionEnd

Section -post
    ; Register for 64-bit first so it has precedence over the 32-bit MPDN
    ${IfNot} $mpdn64_root == ""
        ${registerExtension} "$mpdn64_root\MediaPlayerDotNet.exe" ".mpl" "MPDN Playlist File"
        ${GetFileVersion}  "$mpdn64_root\MediaPlayerDotNet.exe" $R0
        WriteRegStr HKLM "SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\MediaPlayerDotNet_x64" "DisplayVersion" "$R0/${VERSION_1}.${VERSION_2}.${VERSION_3}"      
    ${EndIf}
    ${IfNot} $mpdn32_root == ""
        ${registerExtension} "$mpdn32_root\MediaPlayerDotNet.exe" ".mpl" "MPDN Playlist File"
        ${GetFileVersion}  "$mpdn32_root\MediaPlayerDotNet.exe" $R0
        WriteRegStr HKLM "SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\MediaPlayerDotNet_x86" "DisplayVersion" "$R0/${VERSION_1}.${VERSION_2}.${VERSION_3}"     
    ${EndIf}
    
SectionEnd

;--------------------------------
;Installer Sections

Function .onInit    
    ${IfNot} ${AtLeastWin7}
        MessageBox MB_OK "Windows 7 and above required"
        Quit
    ${EndIf}
    
    System::Call 'kernel32::CreateMutex(i 0, i 0, t "MpdnExtensionsInstaller") ?e'
    Pop $R0
    StrCmp $R0 0 +3
        MessageBox MB_OK "The installer is already running."
        Abort
    
    ${GetParameters} $R0
    ClearErrors
    
    ; Advanced Installer Variables
        ${GetOptions} '$R0' '/ARCH=' $R1
        StrCmp $R1 '' 0 +3
        StrCpy $playerArchitecture ''
        Goto +2
        StrCpy $playerArchitecture $R1
        
        ${GetOptions} '$R0' '/INSTALLER=' $R1
        StrCmp $R1 '' 0 +3
        StrCpy $playerInstallerPath ''
        Goto +2
        StrCpy $playerInstallerPath $R1
    ; END Advanced Installer Variables
    
    !insertmacro SelectSection ${SecMPDNExtensions32}
    !insertmacro SelectSection ${SecMPDNExtensions64}
    
    !insertmacro MULTIUSER_INIT
    SetShellVarContext all
    
    ${If} ${RunningX64}
        SetRegView 64
    ${EndIf}
    
    ReadRegStr $R0 HKLM "SOFTWARE\${MPDN_REGNAME}_x86" ""
    StrCpy $mpdn32_root "$R0"
    ReadRegStr $R0 HKLM "SOFTWARE\${MPDN_REGNAME}_x64" ""
    StrCpy $mpdn64_root "$R0"

    StrCpy $R0 "$mpdn32_root"
    StrCpy $R1 "$mpdn64_root"
    
    ${IfNot} "$playerArchitecture$playerInstallerPath" == ""
        ${If} $playerArchitecture == "x64"
            SectionSetText ${SecMPDNExtensions32} ""
            !insertmacro UnselectSection ${SecMPDNExtensions32}
        ${Else}
            SectionSetText ${SecMPDNExtensions64} ""
            !insertmacro UnselectSection ${SecMPDNExtensions64}
        ${EndIf}
        !insertmacro SelectSection ${SecInstallPlayer}
        
    ${Else}
            
        StrCmp $R0 "" 0 check64
            SectionSetText ${SecMPDNExtensions32} ""
            !insertmacro UnselectSection ${SecMPDNExtensions32}
    check64:
        StrCmp $R1 "" 0 done
            SectionSetText ${SecMPDNExtensions64} ""
            !insertmacro UnselectSection ${SecMPDNExtensions64}
    done:

        StrCmp "$R0$R1" "" 0 +3
            MessageBox MB_OK "Unable to find any installations of MPDN.$\r$\n$\r$\nPlease install MPDN first!"
            Abort
    ${EndIf}

FunctionEnd

Function Welcome.show
    ${NSD_CreateCheckbox} 120u -18u 50% 12u "Perform a clean install (use with care)"
    Pop $doCleanInstall
    SetCtlColors $doCleanInstall "" ${MUI_BGCOLOR}
FunctionEnd

Function Welcome.leave
    ${NSD_GetState} $doCleanInstall $0
    ${If} $0 <> 0
        StrCpy $doCleanInstall "1"
    ${Else}
        StrCpy $doCleanInstall "0"
    ${EndIf}
FunctionEnd

Function un.onInit
    ClearErrors
    !insertmacro MULTIUSER_UNINIT
    SetShellVarContext all
    ${If} ${RunningX64}
        SetRegView 64
    ${EndIf}
FunctionEnd


;--------------------------------
;Uninstaller Sections
Function un.includeUninstall
    DetailPrint "Remove un-modified files"
    !include UnInstallLog.log
FunctionEnd

Section "Uninstall"

    ${GetParent} $INSTDIR $R0
    StrCpy $mpdn_root "$R0"
    Call un.includeUninstall
    Delete "$INSTDIR\Uninstall.exe"
SectionEnd

;--------------------------------
;Descriptions

!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
    !insertmacro MUI_DESCRIPTION_TEXT ${SecMPDNExtensions32} $(DESC_SecMPDNExtensions32)
    !insertmacro MUI_DESCRIPTION_TEXT ${SecMPDNExtensions64} $(DESC_SecMPDNExtensions64)
!insertmacro MUI_FUNCTION_DESCRIPTION_END
