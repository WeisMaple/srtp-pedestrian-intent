param([int]$Port=8765)
$ErrorActionPreference='Stop'
$taskPython='F:\Anaconda\envs\d2l\python.exe'
if(-not(Test-Path -LiteralPath $taskPython)){$taskPython='python'}
$taskRuntime='D:\SRTP_modern_20261027'
if(Test-Path -LiteralPath $taskRuntime){$env:SRTP_MODERN_HOME=$taskRuntime}
$taskFfmpeg='D:\Applications\格式工厂\新建文件夹\FormatFactory'
if(Test-Path -LiteralPath (Join-Path $taskFfmpeg 'ffmpeg.exe')){$env:PATH=$taskFfmpeg+';'+$env:PATH}
$taskData=Join-Path $PSScriptRoot 'artifacts\web\jobs'
if(Test-Path -LiteralPath 'D:\'){$taskData='D:\SRTP_presentation_20261102\jobs'}
& $taskPython -s (Join-Path $PSScriptRoot 'run_web_demo.py') --port $Port --data-dir $taskData
if($LASTEXITCODE -ne 0){throw "Web demo failed: $LASTEXITCODE"}
