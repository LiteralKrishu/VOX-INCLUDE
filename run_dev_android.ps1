# VOX-INCLUDE Android Development Startup Script

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "   VOX-INCLUDE ANDROID DEV ENVIRONMENT STARTUP " -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# 1. Start Backend Server in a new window
Write-Host "1. Starting FastAPI Backend (Port 8000)..." -ForegroundColor Yellow
$backendCommand = "& '.\venv\Scripts\Activate.ps1'; uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "$backendCommand"

# Wait for server to initialize
Write-Host "   Waiting 5 seconds for backend to warm up..." -ForegroundColor DarkGray
Start-Sleep -Seconds 5

# 2. Check for connected devices
Write-Host "2. Checking for connected Android devices..." -ForegroundColor Yellow
# 2. Ensure a connected Android device or emulator
Write-Host "2. Ensuring an Android device/emulator is available..." -ForegroundColor Yellow

function Get-FlutterDevicesOutput {
    return & flutter devices 2>&1
}

while ($true) {
    $devicesOutput = Get-FlutterDevicesOutput
    Write-Host $devicesOutput -ForegroundColor Gray
    if ($devicesOutput -match "No devices detected|No devices|0 connected|No connected devices") {
        Write-Host "No Android device/emulator detected." -ForegroundColor Red
        Write-Host "Options: (E)mulator  (C)onnect device and retry  (R)etry  (Q)uit" -ForegroundColor Yellow
        $choice = Read-Host "Choose an option [E/C/R/Q]"
        switch ($choice.ToUpper()) {
            "E" {
                $sdkRoot = $env:ANDROID_SDK_ROOT
                if (-not $sdkRoot) { $sdkRoot = $env:ANDROID_HOME }
                if (-not $sdkRoot) {
                    Write-Host "ANDROID_SDK_ROOT or ANDROID_HOME not set. Cannot list AVDs." -ForegroundColor Red
                    Write-Host "Open Android Studio > AVD Manager to start an emulator, then press Enter to retry." -ForegroundColor Yellow
                    Read-Host "Press Enter to continue"
                    continue
                }
                $emulatorPath = Join-Path $sdkRoot "emulator\emulator.exe"
                if (-not (Test-Path $emulatorPath)) {
                    Write-Host "Emulator binary not found at $emulatorPath" -ForegroundColor Red
                    Write-Host "Ensure Android SDK is installed and emulator is available." -ForegroundColor Yellow
                    Read-Host "Press Enter to continue"
                    continue
                }
                $avds = & $emulatorPath -list-avds 2>&1
                if (-not $avds) {
                    Write-Host "No AVDs found. Create one in Android Studio AVD Manager." -ForegroundColor Red
                    Read-Host "Press Enter to continue"
                    continue
                }
                Write-Host "Available AVDs:" -ForegroundColor Cyan
                $i=0
                foreach ($avd in $avds) {
                    $i++; Write-Host "[$i] $avd"
                }
                $selection = Read-Host "Enter number to start (or press Enter to start first)"
                if (-not $selection) { $selection = 1 }
                $index = [int]$selection
                if ($index -lt 1 -or $index -gt $avds.Length) { $index = 1 }
                $selectedAvd = $avds[$index-1]
                Write-Host "Starting AVD: $selectedAvd" -ForegroundColor Green
                Start-Process -NoNewWindow -FilePath $emulatorPath -ArgumentList "-avd", $selectedAvd
                Write-Host "Waiting 10 seconds for emulator to start..." -ForegroundColor DarkGray
                Start-Sleep -Seconds 10
                continue
            }
            "C" {
                Write-Host "Please connect your Android device now (enable USB debugging), then press Enter to retry." -ForegroundColor Yellow
                Read-Host "Press Enter when device is connected"
                continue
            }
            "R" {
                continue
            }
            "Q" {
                Write-Host "Exiting." -ForegroundColor Red
                Exit
            }
            default {
                Write-Host "Unknown option. Retrying..." -ForegroundColor Yellow
                continue
            }
        }
    } else {
        Write-Host "Device/emulator found. Proceeding to run the Flutter app." -ForegroundColor Green
        break
    }
}

# 3. Start Flutter App on Android
Write-Host "3. Starting Flutter App on Android..." -ForegroundColor Yellow
Set-Location "mobile_app"

# 'flutter run' automatically picks the connected device/emulator
# Use -v to see verbose output if needed
flutter run

# Note: The backend window remains open after Flutter closes.
Write-Host "Frontend closed. Backend is still running in the other window." -ForegroundColor Green
