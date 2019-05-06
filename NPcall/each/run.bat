@echo off
FOR %%A IN (%*) DO CALL :loopbody %%A
pause
GOTO :EOF

:loopbody
ECHO %1
SET output=%1\np-results
IF NOT EXIST %output% MKDIR %output%
python C:\Users\CenUser\Desktop\NP\NP.py %1 2 4 C:\Users\CenUser\Desktop\NP\annotation_files %output%
GOTO :EOF