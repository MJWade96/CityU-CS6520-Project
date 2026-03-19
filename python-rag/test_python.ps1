Write-Host "Testing Python..."
$version = python --version 2>&1
Write-Host "Python version: $version"

Write-Host "`nRunning minimal_test.py..."
$result = python minimal_test.py 2>&1
Write-Host "Result:"
Write-Host $result

$result | Out-File -FilePath test_python_output.txt -Encoding utf8
Write-Host "`nOutput saved to test_python_output.txt"
