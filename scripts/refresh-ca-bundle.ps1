<#
.SYNOPSIS
    Rebuilds the local CA bundle that lets dev tooling trust an intercepting
    TLS proxy (Avast Web/Mail Shield on this machine).

.DESCRIPTION
    Avast's HTTPS scanning re-signs every TLS connection with its own root.
    Windows trusts that root, so browsers are fine, but tools carrying their
    own CA store are not: uv's rustls fails with `UnknownIssuer`, and Linux
    containers fail with `CERTIFICATE_VERIFY_FAILED`.

    The fix is one bundle holding the Mozilla roots plus the proxy root, with
    SSL_CERT_FILE and friends pointing at it. Avast regenerates its root on
    some updates, which silently breaks the bundle; this script rebuilds it.

    Mozilla roots come from the local certifi package, not a download, because
    a stale bundle means TLS is already broken and a download would fail.

.EXAMPLE
    pwsh -File scripts/refresh-ca-bundle.ps1

.EXAMPLE
    pwsh -File scripts/refresh-ca-bundle.ps1 -Verify
    Only checks whether the current bundle covers the installed proxy roots.
#>
[CmdletBinding()]
param(
    # Where the combined bundle lives. The env vars below point here.
    [string]$BundlePath = (Join-Path $HOME '.ssl\cacert.pem'),

    # Subject substring identifying the proxy's root in the Windows store.
    [string]$ProxySubjectMatch = 'Avast',

    # Report only; make no changes.
    [switch]$Verify,

    # Rebuild even when the bundle already covers every proxy root.
    [switch]$Force,

    # Build the bundle but leave the user env vars alone. For testing a rebuild
    # against a throwaway -BundlePath without repointing the real setup at it.
    [switch]$NoEnv
)

$ErrorActionPreference = 'Stop'

function Get-ProxyRoots {
    # Both stores: Avast has been seen to install into LocalMachine and
    # CurrentUser, and the two copies are not always the same generation.
    $roots = @()
    foreach ($store in 'Cert:\LocalMachine\Root', 'Cert:\CurrentUser\Root') {
        if (Test-Path $store) {
            $roots += Get-ChildItem $store | Where-Object { $_.Subject -like "*$ProxySubjectMatch*" }
        }
    }
    # De-duplicate across stores by thumbprint.
    $roots | Sort-Object Thumbprint -Unique
}

function Get-BundleThumbprints {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return @() }
    $text = Get-Content -Raw -Path $Path
    $found = @()
    foreach ($m in [regex]::Matches($text, '(?s)-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----')) {
        $b64 = ($m.Value -replace '-----[A-Z ]+-----', '') -replace '\s', ''
        try {
            $cert = [System.Security.Cryptography.X509Certificates.X509Certificate2]::new([Convert]::FromBase64String($b64))
            $found += $cert.Thumbprint
        } catch {
            # A malformed block should not abort the audit of the rest.
            Write-Verbose "Skipped an unparseable certificate block: $($_.Exception.Message)"
        }
    }
    $found
}

function Get-CertifiPath {
    # Prefer the project venv, fall back to whatever python is on PATH.
    $candidates = @()
    $venvPython = Join-Path (Split-Path -Parent $PSScriptRoot) '.venv\Scripts\python.exe'
    if (Test-Path $venvPython) { $candidates += $venvPython }
    $onPath = (Get-Command python -ErrorAction SilentlyContinue).Source
    if ($onPath) { $candidates += $onPath }

    foreach ($py in $candidates) {
        # Ignore SSL_CERT_FILE here: if the bundle is stale, importing certifi
        # still works, but we want certifi's own path, not the current override.
        $out = & $py -I -c "import certifi; print(certifi.where())" 2>$null
        if ($LASTEXITCODE -eq 0 -and $out -and (Test-Path $out)) { return $out }
    }
    throw "Could not locate certifi's bundle. Install certifi (uv sync) and retry."
}

# --- audit ---------------------------------------------------------------

$proxyRoots = Get-ProxyRoots
if (-not $proxyRoots) {
    Write-Host "No root matching '$ProxySubjectMatch' is installed. Nothing to inject." -ForegroundColor Yellow
    Write-Host "If the proxy was uninstalled, you can drop the bundle and unset SSL_CERT_FILE."
    return
}

Write-Host "Proxy roots installed in the Windows store:" -ForegroundColor Cyan
foreach ($r in $proxyRoots) {
    Write-Host ("  {0}  expires {1:yyyy-MM-dd}" -f $r.Thumbprint, $r.NotAfter)
}

$inBundle = Get-BundleThumbprints -Path $BundlePath
$missing  = $proxyRoots | Where-Object { $inBundle -notcontains $_.Thumbprint }

if (Test-Path $BundlePath) {
    Write-Host "Current bundle: $BundlePath ($($inBundle.Count) certificates)" -ForegroundColor Cyan
} else {
    Write-Host "Current bundle: absent" -ForegroundColor Yellow
}

if (-not $missing -and -not $Force) {
    Write-Host "Bundle already carries every installed proxy root. No rebuild needed." -ForegroundColor Green
    if (-not $Verify) { Write-Host "Pass -Force to rebuild anyway." }
    return
}

foreach ($m in $missing) {
    Write-Host "  MISSING from bundle: $($m.Thumbprint)" -ForegroundColor Red
}

if ($Verify) {
    Write-Host "-Verify given, so stopping without changes. Re-run without it to rebuild." -ForegroundColor Yellow
    exit 1
}

# --- rebuild -------------------------------------------------------------

$certifiPath = Get-CertifiPath
Write-Host "Mozilla roots from: $certifiPath" -ForegroundColor Cyan

$bundleDir = Split-Path -Parent $BundlePath
if (-not (Test-Path $bundleDir)) { New-Item -ItemType Directory -Path $bundleDir -Force | Out-Null }

if (Test-Path $BundlePath) {
    $backup = "$BundlePath.bak-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    Copy-Item $BundlePath $backup
    Write-Host "Backed up previous bundle to $backup"
}

$sb = [System.Text.StringBuilder]::new()
[void]$sb.AppendLine((Get-Content -Raw -Path $certifiPath).TrimEnd())

foreach ($r in $proxyRoots) {
    $b64 = [Convert]::ToBase64String($r.RawData, 'InsertLineBreaks')
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("# $($r.Subject)")
    [void]$sb.AppendLine("# SHA1 $($r.Thumbprint), injected $(Get-Date -Format 'yyyy-MM-dd')")
    [void]$sb.AppendLine("-----BEGIN CERTIFICATE-----")
    [void]$sb.AppendLine($b64)
    [void]$sb.AppendLine("-----END CERTIFICATE-----")
}

# ASCII, LF: OpenSSL and rustls both reject a UTF-8 BOM at the head of a bundle.
$text = $sb.ToString() -replace "`r`n", "`n"
[System.IO.File]::WriteAllText($BundlePath, $text, (New-Object System.Text.UTF8Encoding $false))

$after = Get-BundleThumbprints -Path $BundlePath
Write-Host "Wrote $BundlePath ($($after.Count) certificates)" -ForegroundColor Green

# --- env vars ------------------------------------------------------------

# User scope, so new shells and new processes inherit it. These cover OpenSSL
# (SSL_CERT_FILE, also read by git and uv), requests, curl and Node.
$vars = 'SSL_CERT_FILE', 'REQUESTS_CA_BUNDLE', 'CURL_CA_BUNDLE', 'NODE_EXTRA_CA_CERTS'
if ($NoEnv) {
    Write-Host "-NoEnv given: leaving user env vars untouched." -ForegroundColor Yellow
    Write-Host "Bundle written to $BundlePath; nothing points at it yet."
    return
}
foreach ($v in $vars) {
    $current = [Environment]::GetEnvironmentVariable($v, 'User')
    if ($current -ne $BundlePath) {
        [Environment]::SetEnvironmentVariable($v, $BundlePath, 'User')
        Write-Host "Set user env $v=$BundlePath" -ForegroundColor Green
    } else {
        Write-Host "User env $v already correct"
    }
    # Also set it in this session, so the verification below uses the new bundle.
    Set-Item -Path "Env:$v" -Value $BundlePath
}

# --- verify --------------------------------------------------------------

Write-Host "`nVerifying..." -ForegroundColor Cyan
$ok = $true

& curl.exe -sS -o NUL -w "curl pypi.org: %{http_code}`n" https://pypi.org/simple/
if ($LASTEXITCODE -ne 0) { $ok = $false; Write-Host "curl failed" -ForegroundColor Red }

$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if ($uv) {
    & $uv pip install --no-cache --refresh --dry-run tabulate==0.9.0 *> $null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "uv reached PyPI over the network: OK" -ForegroundColor Green
    } else {
        $ok = $false
        Write-Host "uv still cannot reach PyPI. Re-run with -Verbose, or check whether Avast rotated its root again." -ForegroundColor Red
    }
}

if ($ok) {
    Write-Host "`nDone. Open a new terminal so the user env vars take effect there." -ForegroundColor Green
    Write-Host "For docker builds, pass the bundle as a secret:"
    Write-Host "  docker build --secret id=ca-bundle,src=$BundlePath ."
    Write-Host "  CS_CA_BUNDLE=$BundlePath docker compose --profile serve build"
} else {
    exit 1
}
