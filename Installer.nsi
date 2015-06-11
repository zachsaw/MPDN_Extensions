; ****************************************************************************
; * Copyright (C) 2002-2010 OpenVPN Technologies, Inc.                       *
; * Copyright (C)      2012 Alon Bar-Lev <alon.barlev@gmail.com>             *
; * Modified for MediaPlayerDotNet by                                        *
; * Copyright (C)      2015 Antoine Aflalo <antoine@aaflalo.me>              *
; * Copyright (C)      2015 Zach Saw <zach.saw@gmail.com>                    *
; * Modified for MPDN Extensions by                                          *
; * Copyright (C)      2015 Zach Saw <zach.saw@gmail.com>                    *
; *  This program is free software; you can redistribute it and/or modify    *
; *  it under the terms of the GNU General Public License version 2          *
; *  as published by the Free Software Foundation.                           *
; ****************************************************************************

; MPDN install script for Windows, using NSIS

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
!insertmacro GetOptions

; Move Files and folder
; Used to move the Extensions
!include 'FileFunc.nsh'
!insertmacro Locate
 
Var /GLOBAL switch_overwrite

; Windows version check
!include WinVer.nsh


;--------------------------------
;Configuration

;General

; Package name as shown in the installer GUI
Name "${PROJECT_NAME} v${VERSION}"

; Installer filename
OutFile "${PROJECT_NAME}_v${VERSION}_Installer.exe"

ShowInstDetails show
ShowUninstDetails show

;--------------------------------
;Modern UI Configuration

; Compile-time constants which we'll need during install
!define MUI_WELCOMEPAGE_TEXT "This wizard will guide you through the installation of ${PROJECT_NAME} v${VERSION}."

!define MUI_COMPONENTSPAGE_TEXT_TOP "Select the components to install/upgrade.  Stop any MPDN processes.  All DLLs are installed locally."

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

LangString DESC_SecMPDNExtensions32 ${LANG_ENGLISH} "Install ${PROJECT_NAME} for MPDN x86."
LangString DESC_SecMPDNExtensions64 ${LANG_ENGLISH} "Install ${PROJECT_NAME} for MPDN x64."

;--------------------------------
;Reserve Files
  
;Things that need to be extracted on first (keep these lines before any File command!)
;Only useful for BZIP2 compression

;ReserveFile "install-whirl.bmp"

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

!macro WriteRegStringIfUndef ROOT SUBKEY KEY VALUE
	Push $R0
	ReadRegStr $R0 "${ROOT}" "${SUBKEY}" "${KEY}"
	${If} $R0 == ""
		WriteRegStr "${ROOT}" "${SUBKEY}" "${KEY}" '${VALUE}'
	${EndIf}
	Pop $R0
!macroend

!macro DelRegKeyIfUnchanged ROOT SUBKEY VALUE
	Push $R0
	ReadRegStr $R0 "${ROOT}" "${SUBKEY}" ""
	${If} $R0 == '${VALUE}'
		DeleteRegKey "${ROOT}" "${SUBKEY}"
	${EndIf}
	Pop $R0
!macroend

;--------------------
;Pre-install section

Section -pre
	${nsProcess::FindProcess} "MediaPlayerDotNet.exe" $R0
	${If} $R0 == 0
		MessageBox MB_YESNO|MB_ICONEXCLAMATION "To perform the specified operation, ${PROJECT_NAME} needs to be closed.$\r$\n$\r$\nClose it now?" /SD IDYES IDNO guiEndNo
		DetailPrint "Closing ${PROJECT_NAME}..."
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


Section /o "MPDN Extensions for MPDN x86" SecMPDNExtensions32

    RMDir /r "$mpdn32_root\Extensions"
    CreateDirectory "$mpdn32_root\Extensions"
    
	SetOverwrite on

	SetOutPath "$mpdn32_root\Extensions"
	
	File /r "TEMP\*.*"
    
SectionEnd

Section /o "MPDN Extensions for MPDN x64" SecMPDNExtensions64

    RMDir /r "$mpdn64_root\Extensions"
    CreateDirectory "$mpdn64_root\Extensions"
    
	SetOverwrite on

	SetOutPath "$mpdn64_root\Extensions"
	
	File /r "TEMP\*.*"
    
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
	StrCpy $switch_overwrite 0
    
    ${GetParameters} $R0
	ClearErrors
	
	!insertmacro SelectByParameter ${SecMPDNExtensions32} SELECT_MPDN32 1
	!insertmacro SelectByParameter ${SecMPDNExtensions64} SELECT_MPDN64 1
	
	!insertmacro MULTIUSER_INIT
	SetShellVarContext all
    
    ${If} ${RunningX64}
		SetRegView 64
    ${EndIf}
    
    ReadRegStr $R0 HKLM "SOFTWARE\${PROJECT_NAME}_x86" ""
    StrCpy $mpdn32_root "$R0"
    ReadRegStr $R0 HKLM "SOFTWARE\${PROJECT_NAME}_x64" ""
    StrCpy $mpdn64_root "$R0"
    
    StrCpy $R0 $mpdn32_root
    MessageBox MB_OK "$R0"
    
    StrCpy $R0 $mpdn64_root
    MessageBox MB_OK "$R0"

    StrCmp $mpdn32 "" 0 check64
        MessageBox MB_OK $mpdn32
        SectionSetText ${SecMPDNExtensions32} ""
        !insertmacro UnselectSection ${SecMPDNExtensions32}
check64:
    StrCmp $mpdn64 "" 0 done
        MessageBox MB_OK $mpdn64
        SectionSetText ${SecMPDNExtensions64} ""
        !insertmacro UnselectSection ${SecMPDNExtensions64}
done:

FunctionEnd

;--------------------------------
;Descriptions

!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
	!insertmacro MUI_DESCRIPTION_TEXT ${SecMPDNExtensions32} $(DESC_SecMPDNExtensions32)
	!insertmacro MUI_DESCRIPTION_TEXT ${SecMPDNExtensions64} $(DESC_SecMPDNExtensions64)
!insertmacro MUI_FUNCTION_DESCRIPTION_END
