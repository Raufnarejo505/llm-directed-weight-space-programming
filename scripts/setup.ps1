# Setup for the WSP / netFound experiment.
# Usage:  pwsh scripts/setup.ps1
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host '1/4 python dependencies'
python -m pip install --quiet torch transformers==4.57.3 datasets numpy safetensors scapy pandas tensorboard psutil matplotlib

Write-Host '2/4 upstream netFound code'
if (-not (Test-Path 'vendor/netFound/src/modules/netFoundModels.py')) {
    git clone --depth 1 https://github.com/SNL-UCSB/netFound vendor/netFound
}

Write-Host '3/4 checkpoint'
if (-not (Test-Path 'checkpoints/original/model.safetensors')) {
    New-Item -ItemType Directory -Force -Path 'checkpoints/original' | Out-Null
    Invoke-WebRequest -Uri 'https://huggingface.co/snlucsb/netFound-small/resolve/main/model.safetensors' `
        -OutFile 'checkpoints/original/model.safetensors' -UseBasicParsing
    Invoke-WebRequest -Uri 'https://huggingface.co/snlucsb/netFound-small/resolve/main/config.json' `
        -OutFile 'checkpoints/original/config.json' -UseBasicParsing
    $h = (Get-FileHash 'checkpoints/original/model.safetensors' -Algorithm SHA256).Hash.ToLower()
    if ($h -ne '9e0080bbf00f6cd54e1394eb7ad484dd3f5b0758404c80d9ac91a77243e06a70') {
        throw "checkpoint hash mismatch: $h"
    }
    Write-Host '    sha256 verified'
}

Write-Host '4/4 dataset'
Write-Host '    download CTU-IDSEVAL-6 (Zenodo 21027042) and unpack pcap.zip into data/ctu/pcap/'
Write-Host '    see data/DATASET.md'
Write-Host ''
Write-Host 'done. next: python tools/build_dataset.py "data/ctu/pcap/pcap/*.pcap" data/flows_tcp.jsonl'
