# OCR the cropped repair time table at high resolution
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

Add-Type -AssemblyName System.Drawing

$languages = [Windows.Media.Ocr.OcrEngine]::AvailableRecognizerLanguages
$lang = $languages | Where-Object { $_.LanguageTag -like 'zh*' } | Select-Object -First 1
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($lang)
Write-Host "Engine language: $($lang.LanguageTag)"

# Process the repair time table crops
$dir = "d:\代码复现"
$pngs = @("report_rt_table.png", "report_rt_row1.png", "report_rt_row2.png", "report_rt_row3.png")
foreach ($name in $pngs) {
    $path = Join-Path $dir $name
    if (-not (Test-Path $path)) { continue }
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "FILE: $name"
    Write-Host "============================================================"

    # Load via System.Drawing, upscale 2x for better OCR
    $img = [System.Drawing.Image]::FromFile($path)
    $w = $img.Width
    $h = $img.Height
    Write-Host "  original size: ${w}x${h}"

    # Upscale 2x
    $newW = $w * 2
    $newH = $h * 2
    $bmp = New-Object System.Drawing.Bitmap($newW, $newH)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $g.DrawImage($img, 0, 0, $newW, $newH)
    $g.Dispose()

    $tmpFile = Join-Path $env:TEMP "ocr_rt.png"
    $bmp.Save($tmpFile, [System.Drawing.Imaging.ImageFormat]::Png)
    $bmp.Dispose()
    $img.Dispose()

    $file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($tmpFile)) ([Windows.Storage.StorageFile])
    $stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
    $decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
    $sbmp = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
    $result = Await ($engine.RecognizeAsync($sbmp)) ([Windows.Media.Ocr.OcrResult])
    Write-Host $result.Text
}
