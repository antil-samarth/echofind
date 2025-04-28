@echo off
setlocal enabledelayedexpansion

:: Check if input directory was provided
if "%~1"=="" (
  echo Usage: %0 input_directory [output_directory]
  echo Example: %0 C:\Music C:\Music\WAV
  exit /b 1
)

:: Set directories
set "INPUT_DIR=%~1"
if "%~2"=="" (
  set "OUTPUT_DIR=%INPUT_DIR%\wav"
) else (
  set "OUTPUT_DIR=%~2"
)

:: Check if input directory exists
if not exist "%INPUT_DIR%" (
  echo Error: Input directory "%INPUT_DIR%" does not exist.
  exit /b 1
)

:: Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" (
  mkdir "%OUTPUT_DIR%"
  echo Created output directory: %OUTPUT_DIR%
)

:: Count MP3 files
set /a FILE_COUNT=0
for %%F in ("%INPUT_DIR%\*.mp3") do set /a FILE_COUNT+=1

if %FILE_COUNT% equ 0 (
  echo No MP3 files found in "%INPUT_DIR%"
  exit /b 0
)

echo Found %FILE_COUNT% MP3 files. Starting conversion...

:: Process each MP3 file
set /a COUNTER=0
for %%F in ("%INPUT_DIR%\*.mp3") do (
  set /a COUNTER+=1
  
  :: Get filename without extension
  set "FILENAME=%%~nF"
  
  :: Replace spaces with underscores in the filename
  set "NEW_FILENAME=!FILENAME: =_!"
  
  echo [!COUNTER!/%FILE_COUNT%] Converting: %%~nxF
  
  :: Convert MP3 to WAV using ffmpeg with renamed file
  ffmpeg -i "%%F" -acodec pcm_s16le -ar 44100 "%OUTPUT_DIR%\!NEW_FILENAME!.wav" -y -loglevel warning
  
  if !ERRORLEVEL! equ 0 (
    echo     Converted to: !NEW_FILENAME!.wav
  ) else (
    echo     Error converting %%~nxF
  )
)

echo.
echo Conversion complete! %COUNTER% files converted to WAV format with spaces replaced by underscores.
echo Files saved to: %OUTPUT_DIR%