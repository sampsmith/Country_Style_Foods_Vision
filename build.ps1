# Country Style Dough Inspector - Windows Build Script
# PowerShell build script for Windows

Write-Host "Country Style Dough Inspector - Build Script" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# Check if git is available
$gitAvailable = $false
try {
    $null = git --version 2>&1
    $gitAvailable = $true
} catch {
    Write-Host "Warning: Git not found. ImGui will need to be cloned manually." -ForegroundColor Yellow
}

# Clone Dear ImGui if not present
if (-not (Test-Path "external\imgui")) {
    if ($gitAvailable) {
        Write-Host "Cloning Dear ImGui..." -ForegroundColor Green
        New-Item -ItemType Directory -Force -Path "external" | Out-Null
        Push-Location "external"
        git clone https://github.com/ocornut/imgui.git
        Push-Location "imgui"
        git checkout v1.90.4  # Stable version
        Pop-Location
        Pop-Location
    } else {
        Write-Host "Error: external\imgui directory not found and git is not available." -ForegroundColor Red
        Write-Host "Please clone ImGui manually:" -ForegroundColor Yellow
        Write-Host "  mkdir external" -ForegroundColor Yellow
        Write-Host "  cd external" -ForegroundColor Yellow
        Write-Host "  git clone https://github.com/ocornut/imgui.git" -ForegroundColor Yellow
        Write-Host "  cd imgui" -ForegroundColor Yellow
        Write-Host "  git checkout v1.90.4" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "ImGui already cloned." -ForegroundColor Green
}

# Check if CMake is available
try {
    $cmakeVersion = cmake --version 2>&1 | Select-Object -First 1
    Write-Host "Found: $cmakeVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: CMake not found. Please install CMake and add it to PATH." -ForegroundColor Red
    Write-Host "Download from: https://cmake.org/download/" -ForegroundColor Yellow
    exit 1
}

# Create build directory
if (-not (Test-Path "build")) {
    New-Item -ItemType Directory -Path "build" | Out-Null
}

Push-Location "build"

# Check for vcpkg toolchain file
$vcpkgToolchain = ""
$vcpkgRoot = $env:VCPKG_ROOT
if (-not $vcpkgRoot) {
    # Try common vcpkg locations
    $commonVcpkgPaths = @(
        "$env:USERPROFILE\vcpkg\scripts\buildsystems\vcpkg.cmake",
        "C:\vcpkg\scripts\buildsystems\vcpkg.cmake",
        "C:\tools\vcpkg\scripts\buildsystems\vcpkg.cmake"
    )
    foreach ($path in $commonVcpkgPaths) {
        if (Test-Path $path) {
            $vcpkgToolchain = $path
            Write-Host "Found vcpkg toolchain: $vcpkgToolchain" -ForegroundColor Green
            break
        }
    }
} else {
    $vcpkgToolchain = "$vcpkgRoot\scripts\buildsystems\vcpkg.cmake"
    if (Test-Path $vcpkgToolchain) {
        Write-Host "Using vcpkg from VCPKG_ROOT: $vcpkgRoot" -ForegroundColor Green
    } else {
        $vcpkgToolchain = ""
    }
}

# Configure with CMake
Write-Host ""
Write-Host "Configuring with CMake..." -ForegroundColor Green
$cmakeArgs = @(
    "..",
    "-DCMAKE_BUILD_TYPE=Release",
    "-G", "Visual Studio 17 2022"  # Use Visual Studio 2022 generator
)

if ($vcpkgToolchain) {
    $cmakeArgs += "-DCMAKE_TOOLCHAIN_FILE=$vcpkgToolchain"
}

# Try different generators if VS2022 is not available
try {
    cmake @cmakeArgs 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed"
    }
} catch {
    Write-Host "Trying Visual Studio 16 2019 generator..." -ForegroundColor Yellow
    $cmakeArgs[-1] = "Visual Studio 16 2019"
    try {
        cmake @cmakeArgs 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Failed"
        }
    } catch {
        Write-Host "Trying MinGW Makefiles generator..." -ForegroundColor Yellow
        $cmakeArgs = @("..", "-DCMAKE_BUILD_TYPE=Release", "-G", "MinGW Makefiles")
        try {
            cmake @cmakeArgs 2>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) {
                throw "Failed"
            }
        } catch {
            Write-Host "Trying default generator..." -ForegroundColor Yellow
            $cmakeArgs = @("..", "-DCMAKE_BUILD_TYPE=Release")
            cmake @cmakeArgs
            if ($LASTEXITCODE -ne 0) {
                Write-Host "CMake configuration failed!" -ForegroundColor Red
                Pop-Location
                exit 1
            }
        }
    }
}

# Build
Write-Host ""
Write-Host "Building..." -ForegroundColor Green

# Check if using Visual Studio generator
$cmakeCache = Get-Content "CMakeCache.txt" -ErrorAction SilentlyContinue
$usingVSGenerator = $cmakeCache | Select-String "CMAKE_GENERATOR:INTERNAL=Visual Studio"

if ($usingVSGenerator) {
    # Use MSBuild for Visual Studio generators
    $solutionFile = Get-ChildItem -Filter "*.sln" | Select-Object -First 1
    if ($solutionFile) {
        # Find MSBuild
        $msbuildPath = "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe"
        if (-not (Test-Path $msbuildPath)) {
            $msbuildPath = "${env:ProgramFiles}\Microsoft Visual Studio\2022\Professional\MSBuild\Current\Bin\MSBuild.exe"
        }
        if (-not (Test-Path $msbuildPath)) {
            $msbuildPath = "${env:ProgramFiles}\Microsoft Visual Studio\2022\Enterprise\MSBuild\Current\Bin\MSBuild.exe"
        }
        if (-not (Test-Path $msbuildPath)) {
            $msbuildPath = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin\MSBuild.exe"
        }
        if (-not (Test-Path $msbuildPath)) {
            Write-Host "MSBuild not found. Trying cmake --build..." -ForegroundColor Yellow
            cmake --build . --config Release
        } else {
            & $msbuildPath $solutionFile.Name /p:Configuration=Release /p:Platform=x64 /m
        }
    } else {
        cmake --build . --config Release
    }
} else {
    # Use cmake --build for other generators
    cmake --build . --config Release
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "1. Ensure OpenCV is installed and CMake can find it" -ForegroundColor Yellow
    Write-Host "2. Ensure GLFW3 is installed and CMake can find it" -ForegroundColor Yellow
    Write-Host "3. If using vcpkg, ensure packages are installed:" -ForegroundColor Yellow
    Write-Host "   vcpkg install opencv:x64-windows glfw3:x64-windows" -ForegroundColor Yellow
    Write-Host "4. Set CMAKE_PREFIX_PATH if libraries are in non-standard locations" -ForegroundColor Yellow
    Pop-Location
    exit 1
}

Pop-Location

Write-Host ""
Write-Host "Build complete!" -ForegroundColor Green
Write-Host ""

# Find the executable
$exePath = ""
$possiblePaths = @(
    "build\Release\country_style_inspector.exe",
    "build\country_style_inspector.exe",
    "build\Release\CountryStyleDoughInspector.exe",
    "build\CountryStyleDoughInspector.exe"
)

foreach ($path in $possiblePaths) {
    if (Test-Path $path) {
        $exePath = $path
        break
    }
}

if ($exePath) {
    Write-Host "Binary: $exePath" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Run with:" -ForegroundColor Yellow
    Write-Host "  .\$exePath" -ForegroundColor White
} else {
    Write-Host "Warning: Could not locate executable. Please check the build directory." -ForegroundColor Yellow
}

Write-Host ""

