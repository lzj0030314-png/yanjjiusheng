# Use Windows built-in OCR to extract text from PNG images
param(
    [string[]]$Images = $args
)

$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | ? { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]

function Await($WinRtTask, $ResultType) {
    $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)
    $netTask = $asTask.Invoke($null, @($WinRtTask))
    $netTask.Wait(-1) | Out-Null
    $netTask.Result
}

[Windows.Media.Ocr.OcrEngine, Windows.Media.Ocr, ContentType=WindowsRuntime] | Out-Null
[Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType=WindowsRuntime] | Out-Null
[Windows.Storage.StorageFile, Windows.Storage, ContentType=WindowsRuntime] | Out-Null

$languages = [Windows.Media.Ocr.OcrEngine]::AvailableRecognizerLanguages
$lang = $languages | Where-Object { $_.LanguageTag -like 'zh*' } | Select-Object -First 1
if (-not $lang) { $lang = $languages | Select-Object -First 1 }
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($lang)

# Get list of files via Get-ChildItem to avoid encoding issues with hardcoded paths
$dir = Get-Item -LiteralPath $PSScriptRoot
Write-Host "Scanning directory: $($dir.FullName)"
$pngs = Get-ChildItem -LiteralPath $dir.FullName -Filter "report_p*.png" | Sort-Object Name
Write-Host "Found $($pngs.Count) report images"
foreach ($img in $pngs) {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Image: $($img.Name)"
    Write-Host "============================================================"
    $file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($img.FullName)) ([Windows.Storage.StorageFile])
    $stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
    $decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
    $bmp = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
    $result = Await ($engine.RecognizeAsync($bmp)) ([Windows.Media.Ocr.OcrResult])
    Write-Host $result.Text
}

# Also try standard pages (just first 10 for now to keep output manageable)
$stdPngs = Get-ChildItem -LiteralPath $dir.FullName -Filter "standard_p*.png" | Sort-Object { [int]($_.BaseName -replace 'standard_p','') } | Select-Object -First 15
Write-Host ""
Write-Host "############################################################"
Write-Host "STANDARD PDF pages (first 15):"
Write-Host "############################################################"
foreach ($img in $stdPngs) {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Image: $($img.Name)"
    Write-Host "============================================================"
    $file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($img.FullName)) ([Windows.Storage.StorageFile])
    $stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
    $decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
    $bmp = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
    $result = Await ($engine.RecognizeAsync($bmp)) ([Windows.Media.Ocr.OcrResult])
    Write-Host $result.Text
}
