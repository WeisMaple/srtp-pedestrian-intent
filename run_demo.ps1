param(
    [Parameter(Mandatory=$true)][string]$InputVideo,
    [string]$OutputVideo
)
$ErrorActionPreference='Stop'
$demoPython='F:\Anaconda\envs\d2l\python.exe'
if(-not(Test-Path -LiteralPath $demoPython)){$demoPython='python'}
$demoArgs=@('-s',(Join-Path $PSScriptRoot 'run_demo.py'),'--input',$InputVideo)
if($OutputVideo){$demoArgs+=@('--output',$OutputVideo)}
& $demoPython @demoArgs
if($LASTEXITCODE -ne 0){throw "Prediction failed with exit code $LASTEXITCODE"}
