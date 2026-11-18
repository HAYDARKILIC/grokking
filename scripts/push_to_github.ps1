<#
.SYNOPSIS
  Publishes this repository as the initial commit of a GitHub repository.

.DESCRIPTION
  Run this once, on Windows. It works in either of two ways:

    * Run it from inside the extracted folder (scripts\push_to_github.ps1).
      It then uses the folder it lives in; no zip is needed.
    * Or put this script next to the downloaded zip (or leave the zip in
      Downloads / on the Desktop) and run it: it finds the archive by its
      contents, whatever it is called, and extracts it into -WorkDir.
      -Zip "C:\path\to\the.zip" names the archive explicitly.

  If the GitHub repository named in -Remote does not exist yet, the script
  creates it (public by default; add -Private for a private one). If it
  already exists it must be empty: no README, no .gitignore, no licence.

  Authentication is handled by Git Credential Manager, which ships with
  Git for Windows: a browser window opens once and asks you to sign in to
  GitHub. The same sign-in is used to create the repository and to push.

  If Windows says "running scripts is disabled on this system", start it as:
      powershell -ExecutionPolicy Bypass -File .\scripts\push_to_github.ps1

.EXAMPLE
  .\scripts\push_to_github.ps1

.EXAMPLE
  .\push_to_github.ps1 -Zip "$env:USERPROFILE\Downloads\05_artifact_repo.zip"

.EXAMPLE
  .\scripts\push_to_github.ps1 -Remote "https://github.com/HAYDARKILIC/grokking.git"
#>

[CmdletBinding()]
param(
    [string] $Zip     = '',
    [string] $WorkDir = (Join-Path $env:USERPROFILE 'Downloads\grokking-repo'),
    [string] $Remote  = 'https://github.com/HAYDARKILIC/grokking.git',
    [string] $Branch  = 'main',
    [string] $Message = 'Code, figures, tables and result summaries for the nonlinear test of two weight-decay identities',
    [string] $Description = 'Code, figures, tables and result summaries for the nonlinear test of two weight-decay identities (grokking)',
    [switch] $Private,
    [switch] $Force,
    [string] $ApiBase = 'https://api.github.com'
)

# Windows PowerShell 5.1 does not offer TLS 1.2 by default; GitHub requires it.
try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12 } catch { }

# 'Continue', not 'Stop': in Windows PowerShell 5.1 (and in the ISE) git's
# normal progress messages on stderr would otherwise be turned into a
# terminating error and abort the script half-way. Every git call below is
# checked through $LASTEXITCODE instead, and cmdlets use -ErrorAction Stop.
$ErrorActionPreference = 'Continue'
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -Scope Global -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

function Step { param($n, $t) Write-Host "`n[$n] $t" -ForegroundColor Cyan }
function Good { param($t) Write-Host "    OK  $t" -ForegroundColor Green }
function Bad  { param($t) Write-Host "    !!  $t" -ForegroundColor Red }
function Fail { param($t) Bad $t; exit 1 }

$expected = @('README.md', 'CITATION.cff', 'LICENSE', 'figures', 'scripts', 'tables')
function Test-RepoRoot { param($dir)
    if (-not $dir -or -not (Test-Path -LiteralPath $dir)) { return $false }
    foreach ($e in $expected) { if (-not (Test-Path -LiteralPath (Join-Path $dir $e))) { return $false } }
    return $true
}

Step 1 'Checking git'
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Bad 'git was not found on PATH.'
    Write-Host '    Install Git for Windows from https://git-scm.com/download/win'
    Write-Host '    then close and reopen PowerShell and run this again.'
    exit 1
}
Good (git --version)

Step 2 'Checking the identity git will record'
$gitName  = git config --global user.name
$gitEmail = git config --global user.email
if (-not $gitName -or -not $gitEmail) {
    Bad 'git has no global user.name / user.email, so the commit would be unattributed.'
    Write-Host '    Set them once, with the address your GitHub account uses:'
    Write-Host '        git config --global user.name  "Haydar Kilic"'
    Write-Host '        git config --global user.email "you@example.com"'
    exit 1
}
Good "$gitName <$gitEmail>"

Step 3 'Finding the files to publish'
$scriptRepo = if ($PSScriptRoot) { Split-Path -Parent $PSScriptRoot } else { $null }
if ($Zip) {
    if (-not (Test-Path -LiteralPath $Zip)) { Fail "Not found: $Zip" }
    $Zip = (Resolve-Path -LiteralPath $Zip).Path
    Good "Archive: $Zip"
    $SourceDir = $null
} elseif (Test-RepoRoot $scriptRepo) {
    $SourceDir = (Resolve-Path -LiteralPath $scriptRepo).Path
    Good "Using the folder this script is in: $SourceDir"
} else {
    # No -Zip and not run from the extracted folder: look for the archive,
    # whatever it is called, next to the script, in Downloads and on the Desktop.
    # A zip qualifies if it holds README.md and CITATION.cff at its root
    # (or inside a single top-level folder). The newest one wins.
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $searchDirs = @($PSScriptRoot, (Get-Location).Path,
                    (Join-Path $env:USERPROFILE 'Downloads'),
                    (Join-Path $env:USERPROFILE 'Desktop')) |
                  Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -Unique
    $candidates = foreach ($d in $searchDirs) {
        Get-ChildItem -LiteralPath $d -Filter *.zip -File -ErrorAction SilentlyContinue | ForEach-Object {
            try {
                $z = [System.IO.Compression.ZipFile]::OpenRead($_.FullName)
                $names = @($z.Entries | ForEach-Object { $_.FullName -replace '\\', '/' })
                $z.Dispose()
                if (($names | Where-Object { $_ -match '^([^/]+/)?README\.md$' }) -and
                    ($names | Where-Object { $_ -match '^([^/]+/)?CITATION\.cff$' })) { $_ }
            } catch { }
        }
    }
    $pick = $candidates | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $pick) {
        Bad 'Could not find the repository zip next to this script, in Downloads or on the Desktop.'
        Write-Host '    Pass it explicitly:  .\push_to_github.ps1 -Zip "C:\path\to\the.zip"'
        exit 1
    }
    $Zip = $pick.FullName
    $SourceDir = $null
    Good "Found archive: $Zip"
}

Step 4 'Preparing the working folder'
if ($SourceDir) {
    # Work directly in the extracted folder: no copy needed.
    $WorkDir = $SourceDir
    if (Test-Path -LiteralPath (Join-Path $WorkDir '.git')) {
        if (-not $Force) {
            Bad "$WorkDir already has a .git folder (probably from an earlier attempt)."
            Write-Host '    Re-run with -Force to discard it and start the history afresh.'
            exit 1
        }
        Remove-Item -LiteralPath (Join-Path $WorkDir '.git') -Recurse -Force -ErrorAction Stop
        Good 'Removed the old .git folder'
    }
} else {
    if (Test-Path -LiteralPath $WorkDir) {
        $existing = @(Get-ChildItem -LiteralPath $WorkDir -Force)
        if ($existing.Count -gt 0 -and -not $Force) {
            Bad "$WorkDir already exists and is not empty."
            Write-Host '    Delete it, choose another with -WorkDir, or re-run with -Force to overwrite.'
            exit 1
        }
        if ($Force) { Remove-Item -LiteralPath $WorkDir -Recurse -Force -ErrorAction Stop }
    }
    New-Item -ItemType Directory -Path $WorkDir -Force -ErrorAction Stop | Out-Null
    Expand-Archive -LiteralPath $Zip -DestinationPath $WorkDir -Force -ErrorAction Stop
    # If the zip wraps everything in one top-level folder, step into it.
    if (-not (Test-RepoRoot $WorkDir)) {
        $inner = @(Get-ChildItem -LiteralPath $WorkDir -Directory -Force)
        if ($inner.Count -eq 1 -and (Test-RepoRoot $inner[0].FullName)) { $WorkDir = $inner[0].FullName }
    }
}
if (-not (Test-RepoRoot $WorkDir)) {
    Fail "'$WorkDir' does not contain $($expected -join ', '). This does not look like the right folder."
}
$n = @(Get-ChildItem -LiteralPath $WorkDir -Recurse -File -Force).Count
Good "$WorkDir  ($n files)"

Step 5 'Checking that nothing exceeds GitHub''s 100 MB file limit'
$big = Get-ChildItem -LiteralPath $WorkDir -Recurse -File -Force | Where-Object { $_.Length -gt 100MB }
if ($big) {
    Bad 'These files are too large to push:'
    $big | ForEach-Object { Write-Host ("      {0:N1} MB  {1}" -f ($_.Length / 1MB), $_.FullName) }
    exit 1
}
Good 'All files are under the limit'

Step 6 'Signing in to GitHub (a browser window may open)'
if ($Remote -notmatch 'github\.com[/:]([^/]+)/([^/]+?)(\.git)?/?$') {
    Fail "Cannot read owner/name from $Remote (expected https://github.com/OWNER/NAME.git)"
}
$Owner = $Matches[1]; $RepoName = $Matches[2]

# Ask Git Credential Manager for the GitHub sign-in; it opens the browser if
# there is none stored yet. The token it returns is used for the GitHub API.
$credQuery = "protocol=https`nhost=github.com`n`n"
$cred = @{}
$credQuery | git credential fill 2>$null | ForEach-Object {
    $i = "$_".IndexOf('='); if ($i -gt 0) { $cred["$_".Substring(0, $i)] = "$_".Substring($i + 1) }
}
if (-not $cred['password']) { Fail 'No GitHub sign-in was obtained (the sign-in window was closed or failed).' }
$apiHeaders = @{ Authorization = "token $($cred['password'])"; Accept = 'application/vnd.github+json'; 'User-Agent' = 'push_to_github.ps1' }
function Get-HttpStatus { param($err) try { [int]$err.Exception.Response.StatusCode } catch { 0 } }
function Forget-Credential {
    "protocol=https`nhost=github.com`nusername=$($cred['username'])`npassword=$($cred['password'])`n`n" | git credential reject 2>$null
}

try {
    $me = Invoke-RestMethod -Uri "$ApiBase/user" -Headers $apiHeaders -ErrorAction Stop
} catch {
    Forget-Credential
    Fail "GitHub rejected the stored sign-in (HTTP $(Get-HttpStatus $_)). It has been cleared; run this script again to sign in afresh."
}
Good "Signed in as $($me.login)"
if ($me.login -ne $Owner) {
    Forget-Credential
    Bad "You are signed in as '$($me.login)', but the repository belongs to '$Owner'."
    Write-Host '    That sign-in has been cleared. Run this script again and, in the browser,'
    Write-Host "    sign in as $Owner (sign out of '$($me.login)' on github.com first if needed)."
    exit 1
}
# Keep the sign-in so the push below does not ask again.
"protocol=https`nhost=github.com`nusername=$($cred['username'])`npassword=$($cred['password'])`n`n" | git credential approve 2>$null

Step 7 "Making sure github.com/$Owner/$RepoName exists"
$exists = $true
try {
    Invoke-RestMethod -Uri "$ApiBase/repos/$Owner/$RepoName" -Headers $apiHeaders -ErrorAction Stop | Out-Null
} catch {
    if ((Get-HttpStatus $_) -eq 404) { $exists = $false } else { Fail "GitHub API error: $($_.Exception.Message)" }
}
if ($exists) {
    Good 'Repository already exists'
} else {
    $body = @{ name = $RepoName; description = $Description; private = [bool]$Private; auto_init = $false } | ConvertTo-Json
    try {
        Invoke-RestMethod -Method Post -Uri "$ApiBase/user/repos" -Headers $apiHeaders -Body $body `
                          -ContentType 'application/json' -ErrorAction Stop | Out-Null
    } catch {
        Bad "GitHub refused to create the repository (HTTP $(Get-HttpStatus $_))."
        Write-Host "    Create it by hand at https://github.com/new (name: $RepoName, nothing ticked),"
        Write-Host '    then run this script again.'
        exit 1
    }
    Good "Created $(if ($Private) { 'private' } else { 'public' }) repository https://github.com/$Owner/$RepoName"
}

$heads = git ls-remote --heads $Remote 2>&1 | ForEach-Object { "$_" }
if ($LASTEXITCODE -ne 0) {
    Bad "GitHub did not answer for $Remote"
    $heads | ForEach-Object { Write-Host "      $_" }
    Write-Host '    Check that the repository exists with exactly this name and owner, and that'
    Write-Host '    you are signed in as that owner. Pass the right one with -Remote "https://github.com/USER/REPO.git"'
    exit 1
}
if ($heads | Where-Object { $_ -match 'refs/heads/' }) {
    Bad 'That repository already has commits (it was probably created with a README, .gitignore or licence).'
    Write-Host '    Delete and re-create it on GitHub WITHOUT any of those boxes ticked, then run this again.'
    exit 1
}
Good 'Repository found and empty'

Push-Location -LiteralPath $WorkDir
try {
    Step 8 'Creating the local repository and the first commit'
    git init -q -b $Branch 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {       # git older than 2.28 has no -b
        git init -q 2>&1 | Out-Null
        git checkout -q -b $Branch 2>&1 | Out-Null
    }
    if (-not (Test-Path -LiteralPath '.git')) { Fail 'git init failed.' }

    git add -A 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { Fail 'git add failed.' }
    $staged = @(git diff --cached --name-only).Count
    if ($staged -eq 0) { Fail 'Nothing was staged. Stopping.' }

    git commit -q -m $Message
    if ($LASTEXITCODE -ne 0) { Fail 'git commit failed (see the message above).' }
    Good "$staged files committed on '$Branch'"

    Step 9 "Pointing the repository at $Remote"
    git remote remove origin 2>&1 | Out-Null
    git remote add origin $Remote
    if ($LASTEXITCODE -ne 0) { Fail 'Could not add the remote.' }
    Good 'remote "origin" added'

    Step 10 'Pushing'
    git push -u origin $Branch
    if ($LASTEXITCODE -ne 0) {
        Bad 'The push failed; git''s own message is printed above.'
        Write-Host '    "rejected / fetch first"  -> the GitHub repository is not empty.'
        Write-Host '    "403" / "Permission denied" -> you are signed in as a different GitHub user.'
        Write-Host '        Clear it in Windows Credential Manager (Windows Credentials > git:https://github.com)'
        Write-Host '        and run this again.'
        Write-Host '    "Repository not found" -> -Remote has the wrong owner or name.'
        exit 1
    }
    Good 'Pushed'
} finally {
    Pop-Location
}

Write-Host "`nDone. The repository is at $($Remote -replace '\.git$','')" -ForegroundColor Green
Write-Host @"

Three things remain, on GitHub itself:
  1. Settings > General: add the description and the topics.
  2. Releases > Draft a new release, tag v1.0.0, so the code has a fixed
     citable state.
  3. Optionally connect Zenodo to mint a DOI, then put that DOI into
     CITATION.cff and into the manuscript's Data and Code Availability
     statement, which currently carries a placeholder.
"@
