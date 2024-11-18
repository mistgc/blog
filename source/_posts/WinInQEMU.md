---
title: Windows 11 in QEMU
subtitle: Trusted Platform Module
date: 2022/9/21
tags: tech
---

![Windows 11 in QEMU](https://img1.imgtp.com/2022/09/21/HsnYHd5d.png)

The Windows 11 needs to check TPM.

There are two methods for resolving the problems.
1. Use 'swtpm' to emulate a Trusted Platform Module(TPM):

    https://wiki.archlinux.org/title/QEMU#Trusted_Platform_Module_emulation

2. Bypass the 'TPM check':

    1). Enter intalling interface.

    2). Press 'Shift + F10' to open the command prompt.

    3). Type 'regedit' in command prompt then open the Windows Register Editor.

    4). Navigate to 'HKEY_LOCAL_MACHINE\SYSTEM\Setup', and right-click on the 'Setup' key and select 'New', then select 'Key'.

    5). Name the new key to 'LabConfig'.

    6). Right-click the 'LabConfig' key and select 'New', then select 'DWORD (32-bit)' and create a value named 'BypassTPMCheck', andd set its data to '1'. With the same steps create the 'BypassRAMCheck' and 'BypassSecureBootCheck' values and set also their data to '1'.

    7). Close the Windows Register Editor and exit command prompt.

    8). Click on the 'Install now' button to proceed to get 'Microsoft Windows 11'.

Reference:
https://blogs.oracle.com/virtualization/post/install-microsoft-windows-11-on-virtualbox
