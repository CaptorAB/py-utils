<#
.SYNOPSIS
    Windows counterpart of Makefile (and of source_me for 'active').

.PARAMETER task
    What to do: 'all', 'install', 'update', 'test', 'lint', 'clean', or 'active'.
    'make' is accepted as an alias of 'install'.
    Defaults to 'active' (the Windows equivalent of 'source source_me').
#>

param (
    [ValidateSet("active", "all", "install", "make", "update", "test", "lint", "clean")]
    [string]$task = "active"
)

$ErrorActionPreference = 'Stop'

# Pin Poetry to the same version as Makefile.
[string]$poetryVersion = '2.4.1'

function Set-ProjectPythonPath {
    if ($env:PYTHONPATH) {
        if (-not ($env:PYTHONPATH -match [regex]::Escape($PWD))) {
            $env:PYTHONPATH = "$PWD;$env:PYTHONPATH"
            Write-Output "`nPYTHONPATH updated to: $env:PYTHONPATH"
        } else {
            Write-Output "`nPYTHONPATH already includes project root."
        }
    } else {
        $env:PYTHONPATH = "$PWD"
        Write-Output "`nPYTHONPATH set to: $env:PYTHONPATH"
    }
}

function Enter-ProjectVenv {
    $activate = Join-Path $PWD 'venv\Scripts\Activate.ps1'
    if (-not (Test-Path $activate)) {
        Throw "Virtual environment not found at .\venv. Run '.\make.ps1 install' first."
    }
    . $activate
    Set-ProjectPythonPath
    Write-Output "`nThe Python used in the '$(Split-Path $env:VIRTUAL_ENV -Leaf)' environment is:"
    python --version
}

Push-Location (Split-Path -Parent $MyInvocation.MyCommand.Definition)

try {
    switch ($task) {
        "active" {
            Enter-ProjectVenv
        }

        { $_ -in "all", "install", "make" } {
            python -m venv .\venv
            $venvPython = Join-Path $PWD 'venv\Scripts\python.exe'
            & $venvPython --version
            & $venvPython -m pip install --upgrade pip
            & $venvPython -m pip install "poetry==$poetryVersion"

            Enter-ProjectVenv
            poetry install --no-root --with dev
            poetry run pre-commit install
            poetry export --output requirements.txt --without-hashes --all-groups
        }

        "update" {
            Enter-ProjectVenv
            poetry update --with dev
            poetry export --output requirements.txt --without-hashes --all-groups
        }

        "test" {
            Enter-ProjectVenv
            poetry run pytest
        }

        "lint" {
            Enter-ProjectVenv
            poetry run ruff format
            poetry run ruff check . --fix --exit-non-zero-on-fix
        }

        "clean" {
            if (Test-Path .\venv\Scripts\Activate.ps1) {
                . .\venv\Scripts\Activate.ps1
                pre-commit uninstall
            }
            if ($env:VIRTUAL_ENV) {
                & "$env:VIRTUAL_ENV\Scripts\deactivate.bat" 2>$null
            }
            if (Test-Path .\venv) {
                Remove-Item .\venv -Recurse -Force
            }
            if (Test-Path 'poetry.lock') {
                Remove-Item 'poetry.lock' -Force
            }
            if (Test-Path 'requirements.txt') {
                Remove-Item 'requirements.txt' -Force
            }
        }

        default {
            Write-Error "Invalid task '$task'. Use all, install, update, test, lint, clean, or active."
            exit 1
        }
    }
} finally {
    Pop-Location
}
