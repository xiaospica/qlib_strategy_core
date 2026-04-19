@echo off
call "F:\Program_Home\Microsoft Visual Studio\VC\Auxiliary\Build\vcvars64.bat"
set DISTUTILS_USE_SDK=1
set MSSdk=1
cd /d F:\Quant\code\qlib_strategy_dev\vendor\qlib_strategy_core\qlib\data\_libs
F:\Program_Home\vnpy\python.exe _build_extensions.py build_ext --inplace
