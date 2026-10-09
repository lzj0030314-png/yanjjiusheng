# -*- coding: utf-8 -*-
"""Crop repair time section from report and OCR with Windows OCR."""
import subprocess, sys, os
from pathlib import Path

# Use PIL to crop
from PIL import Image

src = r'd:\代码复现\report_hq_p1.png'
img = Image.open(src)
print(f'Image size: {img.size}')

# The repair time section is on the bottom-left of page 1
# Full image is 2382x3368. Crop bottom-left quadrant
w, h = img.size
# Bottom-left quadrant
crop_bl = img.crop((0, h//2, w//2, h))
crop_bl.save(r'd:\代码复现\report_repair_time.png')
print(f'Saved crop: {crop_bl.size}')

# Also crop just the repair time table area (lower portion)
# Based on the OCR, the repair time section is in the lower-middle
crop_rt = img.crop((0, int(h*0.55), int(w*0.75), int(h*0.85)))
crop_rt.save(r'd:\代码复现\report_repair_time2.png')
print(f'Saved crop2: {crop_rt.size}')

# Now use Windows OCR via PowerShell
ps_script = r'''
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | ? { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]

function Await($WinRtTask, $ResultType) {
    $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)
    $netTask = $asTask.Invoke($null, @($WinRtTask))
    $netTask.Wait(-1) | Out-Null
    $netTask.Result
}

function RunOcr($imagePath) {
    Add-Type -AssemblyName Windows.Security
    $stream = [Windows.Storage.Streams.RandomAccessStreamReference]::CreateFromFile($imagePath)
    $streamRef = [Windows.Storage.Streams.IRandomAccessStreamReference]$stream
    $asyncOp = [Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync((Await $stream.OpenReadAsync() ([Windows.Storage.Streams.IRandomAccessStream])))
    # Simplified: use OcrEngine directly on the image file
    $file = Get-Item $imagePath
    $storageFile = [Windows.Storage.StorageFile]::GetFileFromPathAsync($file.FullName)
    # Actually, let's use a simpler approach
    Write-Output "OCR not directly available in PS, using alternative"
}

# Alternative: use the .NET System.Speech or just use the image directly
# Actually, let's try Windows.Media.Ocr via PowerShell
[Windows.Media.Ocr.OcrEngine, Windows.Media.Ocr, ContentType=WindowsRuntime] | Out-Null
[Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType=WindowsRuntime] | Out-Null
[Windows.Storage.StorageFile, Windows.Storage, ContentType=WindowsRuntime] | Out-Null
[Windows.Storage.Streams.RandomAccessStreamReference, Windows.Storage.Streams, ContentType=WindowsRuntime] | Out-Null

function Invoke-WinRTOcr {
    param([string]$ImagePath)
    
    $file = [Windows.Storage.StorageFile]::GetFileFromPathAsync($ImagePath)
    # Wait for the async operation
    $filePath = $ImagePath
    $uri = [System.Uri]$filePath
    
    # Use BitmapDecoder
    $ras = [Windows.Storage.Streams.RandomAccessStreamReference]::CreateFromUri($uri)
    
    # Actually this is getting too complex. Let me use a Python-based OCR approach
    Write-Output "Trying Python approach instead"
}

Write-Output "PowerShell OCR approach - skipping complex WinRT"
'''

# Instead, let's try using the pytesseract or easyocr if available
# Or better: let's just crop and save smaller sections and view them in browser

# Crop the specific repair time table rows
# Based on the high-res image layout, the table is likely around y=2000-2800
crop_table = img.crop((0, 1900, 1800, 2900))
crop_table.save(r'd:\代码复现\report_rt_table.png')
print(f'Saved table crop: {crop_table.size}')

# Also try to get individual rows
# Row 1 (floor 1): around y=2200
# Row 2 (floor 2): around y=2350
# Row 3 (floor 3): around y=2500
for i, (y1, y2, label) in enumerate([(2150, 2280, 'row1'), (2280, 2410, 'row2'), (2410, 2540, 'row3')]):
    crop = img.crop((100, y1, 1700, y2))
    crop.save(rf'd:\代码复现\report_rt_{label}.png')
    print(f'Saved {label}: {crop.size}')
