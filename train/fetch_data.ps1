# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
# Download and unpack the training / evaluation data listed in docs/DATA.md into $Root, log sizes and SHA-256 hashes.
#   powershell -ExecutionPolicy Bypass -File fetch_data.ps1 -Root G:\cosgc\data
# Skips archives that are already downloaded and unpacked. HTTPS only; a certificate error stops that download.
param([string]$Root = "G:\cosgc\data")
$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force $Root | Out-Null
Set-Location $Root
$log = Join-Path $Root "fetch_log.tsv"
if (-not (Test-Path $log)) { "date`tname`turl`tbytes`tsha256`tstatus" | Out-File -Encoding utf8 $log }

$items = @(
  # name, url, marker (path that exists once unpacked)
  @("librispeech-train-clean-100", "https://www.openslr.org/resources/12/train-clean-100.tar.gz", "LibriSpeech\train-clean-100"),
  @("librispeech-train-clean-360", "https://www.openslr.org/resources/12/train-clean-360.tar.gz", "LibriSpeech\train-clean-360"),
  @("librispeech-dev-clean",      "https://www.openslr.org/resources/12/dev-clean.tar.gz",       "LibriSpeech\dev-clean"),
  @("librispeech-test-clean",      "https://www.openslr.org/resources/12/test-clean.tar.gz",      "LibriSpeech\test-clean"),
  @("librispeech-test-other",      "https://www.openslr.org/resources/12/test-other.tar.gz",      "LibriSpeech\test-other"),
  # TORGO archives unpack into one folder per speaker
  @("torgo-F",  "https://www.cs.toronto.edu/~complingweb/data/TORGO/F.tar.bz2",  "TORGO\F01"),
  @("torgo-FC", "https://www.cs.toronto.edu/~complingweb/data/TORGO/FC.tar.bz2", "TORGO\FC01"),
  @("torgo-M",  "https://www.cs.toronto.edu/~complingweb/data/TORGO/M.tar.bz2",  "TORGO\M01"),
  @("torgo-MC", "https://www.cs.toronto.edu/~complingweb/data/TORGO/MC.tar.bz2", "TORGO\MC01"),
  @("musan",    "https://www.openslr.org/resources/17/musan.tar.gz",            "musan")
)

foreach ($it in $items) {
  $name, $url, $marker = $it
  if (Test-Path (Join-Path $Root $marker)) { Write-Output "$name already unpacked"; continue }
  $file = Join-Path $Root ([IO.Path]::GetFileName($url))
  if ($name -like "torgo-*") { $file = Join-Path $Root ("torgo_" + [IO.Path]::GetFileName($url)) }
  Write-Output "$(Get-Date -Format s) downloading $name"
  # a transfer slower than 10 kB/s for 60 s counts as stalled; OpenSLR files fall back to the ELDA mirror
  $urls = @($url)
  if ($url -like "https://www.openslr.org/*") { $urls += $url.Replace("https://www.openslr.org", "https://openslr.elda.org") }
  foreach ($u in $urls) {
    for ($try = 0; $try -lt 3; $try++) {
      curl.exe -sSfL --speed-limit 10000 --speed-time 60 -C - -o $file $u
      if ($LASTEXITCODE -eq 0) { break }
      Write-Output "  curl $LASTEXITCODE from $u, retrying"
    }
    if ($LASTEXITCODE -eq 0) { $url = $u; break }
  }
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path $file)) {
    "$(Get-Date -Format s)`t$name`t$url`t0`t-`tDOWNLOAD FAILED (curl $LASTEXITCODE)" | Out-File -Append -Encoding utf8 $log
    continue
  }
  $bytes = (Get-Item $file).Length
  $sha = (Get-FileHash -Algorithm SHA256 $file).Hash
  $dest = $Root
  if ($name -like "torgo-*") { $dest = Join-Path $Root "TORGO"; New-Item -ItemType Directory -Force $dest | Out-Null }
  tar -xf $file -C $dest
  $status = if ($LASTEXITCODE -eq 0) { "ok" } else { "UNPACK FAILED (tar $LASTEXITCODE)" }
  "$(Get-Date -Format s)`t$name`t$url`t$bytes`t$sha`t$status" | Out-File -Append -Encoding utf8 $log
  Write-Output "$name $bytes bytes $status"
}
Write-Output "done"
