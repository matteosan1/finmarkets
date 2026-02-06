# 'set -e' in PowerShell
$ErrorActionPreference = "Stop"

$function_tests = Get-ChildItem -Name "test_*.py"

foreach ($i in $function_tests) {
    # basename in PowerShell
    $bname = Split-Path $i -Leaf
    
    Write-Host "Running $bname..."
    
    python "./$i"
}