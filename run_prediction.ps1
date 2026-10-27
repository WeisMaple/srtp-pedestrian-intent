param(
    [Parameter(Mandatory=$true)][string]$InputVideo,
    [string]$OutputVideo
)
$ErrorActionPreference='Stop'
$pythonPath='F:\Anaconda\envs\d2l\python.exe'
$runnerPath=Join-Path $PSScriptRoot 'run_prediction.py'
$runnerArgs=@('-s',$runnerPath,'--input',$InputVideo)
if($OutputVideo){$runnerArgs+=@('--output',$OutputVideo)}
& $pythonPath @runnerArgs
if($LASTEXITCODE -ne 0){throw "Prediction failed with exit code $LASTEXITCODE"}