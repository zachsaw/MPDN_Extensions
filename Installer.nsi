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

!addplugindir ./
!include "nsProcess.nsh"

; x64.nsh for architecture detection
!include "x64.nsh"

; Read the command-line parameters
!insertmacro GetParameters

!getdllversion "TEMP\Extensions\Mpdn.Extensions.dll" VERSION_

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

!define MUI_COMPONENTSPAGE_TEXT_TOP "Select the components to install/upgrade.  Stop any MPDN processes.$\r$\n*** WARNING ***  Existing extensions will be removed and replaced!"

!define MUI_COMPONENTSPAGE_SMALLDESC

!define MUI_FINISHPAGE_NOAUTOCLOSE
!define MUI_ABORTWARNING
!define MUI_HEADERIMAGE

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

Var /GLOBAL mpdn32_root
Var /GLOBAL mpdn64_root

;--------------------------------
;Languages
 
!insertmacro MUI_LANGUAGE "English"
  
;--------------------------------
;Language Strings

LangString DESC_SecMPDNExtensions32 ${LANG_ENGLISH} "Install ${PROJECT_NAME} for MPDN 32-bit Edition."
LangString DESC_SecMPDNExtensions64 ${LANG_ENGLISH} "Install ${PROJECT_NAME} for MPDN 64-bit Edition."

;--------------------------------
;Macros

!macro SelectByParameter SECT PARAMETER DEFAULT
	${GetOptions} $R0 "/${PARAMETER}=" $0
	${If} ${DEFAULT} == 0
		${If} $0 == 1
			!insertmacro SelectSection ${SECT}
		${EndIf}
	${Else}
		${If} $0 != 0
			!insertmacro SelectSection ${SECT}
		${EndIf}
	${EndIf}
!macroend

!macro InstallExtensions path
    RMDir /r "${path}\Extensions"
    
	SetOverwrite on

	SetOutPath "${path}"
	
	File /r "TEMP\*.*"
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

Section /o "Extensions for MPDN x86" SecMPDNExtensions32

    !insertmacro InstallExtensions "$mpdn32_root"
    
SectionEnd

Section /o "Extensions for MPDN x64" SecMPDNExtensions64

    !insertmacro InstallExtensions "$mpdn64_root"
    
SectionEnd

;--------------------------------
;Installer Sections

Function .onInit	
	${IfNot} ${AtLeastWin7}
		MessageBox MB_OK "Windows 7 and above required"
		Quit
	${EndIf}
	
	System::Call 'kernel32::CreateMutex(i 0, i 0, t "myMutex") ?e'
	Pop $R0
	StrCmp $R0 0 +3
		MessageBox MB_OK "The installer is already running."
		Abort
    
    ${GetParameters} $R0
	ClearErrors
	
	!insertmacro SelectByParameter ${SecMPDNExtensions32} SELECT_MPDN32 1
	!insertmacro SelectByParameter ${SecMPDNExtensions64} SELECT_MPDN64 1
	
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

FunctionEnd

;--------------------------------
;Descriptions

!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
	!insertmacro MUI_DESCRIPTION_TEXT ${SecMPDNExtensions32} $(DESC_SecMPDNExtensions32)
	!insertmacro MUI_DESCRIPTION_TEXT ${SecMPDNExtensions64} $(DESC_SecMPDNExtensions64)
!insertmacro MUI_FUNCTION_DESCRIPTION_END
