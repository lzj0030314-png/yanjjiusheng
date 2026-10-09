# High-quality OCR on the high-resolution report images
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

# Process each high-quality report page
$dir = "d:\代码复现"
$pngs = @("report_hq_p1.png", "report_hq_p2.png", "report_hq_p3.png")
foreach ($name in $pngs) {
    $path = Join-Path $dir $name
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "FILE: $name"
    Write-Host "============================================================"

    # Load via System.Drawing to allow cropping into top-half/bottom-half for better OCR
    $img = [System.Drawing.Image]::FromFile($path)
    $w = $img.Width
    $h = $img.Height
    Write-Host "  full size: ${w}x${h}"

    # Crop into 4 quadrants for finer OCR
    $quadrants = @(
        @{ Name = "TL"; X = 0;       Y = 0;       W = [int]($w); H = [int]($h/2) },
        @{ Name = "BL"; X = 0;       Y = [int]($h/2); W = [int]($w); H = [int]($h - $h/2) }
    )

    foreach ($q in $quadrants) {
        $bmp = New-Object System.Drawing.Bitmap($q.W, $q.H)
        $g = [System.Drawing.Graphics]::FromImage($bmp)
        $g.DrawImage($img, (New-Object System.Drawing.Rectangle(0, 0, $q.W, $q.H)), (New-Object System.Drawing.Rectangle($q.X, $q.Y, $q.W, $q.H)), [System.Drawing.GraphicsUnit]::Pixel)
        $g.Dispose()

        $tmpFile = Join-Path $env:TEMP "ocr_quad.png"
        $bmp.Save($tmpFile, [System.Drawing.Imaging.ImageFormat]::Png)
        $bmp.Dispose()

        $file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($tmpFile)) ([Windows.Storage.StorageFile])
        $stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
        $decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
        $sbmp = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
        $result = Await ($engine.RecognizeAsync($sbmp)) ([Windows.Media.Ocr.OcrResult])
        Write-Host ""
        Write-Host "---- Quadrant $($q.Name) ----"
        Write-Host $result.Text
    }
    $img.Dispose()
}
