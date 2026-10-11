# GraphRAG — Neo4j Local Launcher
# Run this in a SEPARATE terminal, BEFORE running: python run.py

$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$Neo4jDir   = Join-Path $ScriptDir "neo4j"
$Neo4jVersion = "5.26.0"

if (-Not (Test-Path "$Neo4jDir\bin\neo4j.bat")) {
    Write-Host "Neo4j not found. Downloading $Neo4jVersion ..." -ForegroundColor Yellow
    $zip = Join-Path $ScriptDir "neo4j.zip"
    $url = "https://dist.neo4j.org/neo4j-community-$Neo4jVersion-windows.zip"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    (New-Object System.Net.WebClient).DownloadFile($url, $zip)
    Expand-Archive -Path $zip -DestinationPath $ScriptDir -Force
    $extracted = Get-ChildItem $ScriptDir -Filter "neo4j-community-*" -Directory | Select-Object -First 1
    if ($extracted) { Rename-Item $extracted.FullName "neo4j" -Force }
    Remove-Item $zip -Force -ErrorAction SilentlyContinue
    Write-Host "Extracted." -ForegroundColor Green

    # Set initial password once on first install
    $env:NEO4J_HOME = $Neo4jDir
    & "$Neo4jDir\bin\neo4j-admin.bat" dbms set-initial-password password 2>&1 | Out-Null

    # Patch config
    $conf = Get-Content "$Neo4jDir\conf\neo4j.conf"
    $conf = $conf -replace '^#?(server\.bolt\.enabled).*',            'server.bolt.enabled=true'
    $conf = $conf -replace '^#?(server\.bolt\.listen_address).*',     'server.bolt.listen_address=0.0.0.0:7687'
    $conf = $conf -replace '^#?(server\.default_listen_address).*',   'server.default_listen_address=0.0.0.0'
    $conf | Set-Content "$Neo4jDir\conf\neo4j.conf"
}

$env:NEO4J_HOME = $Neo4jDir
Write-Host ""
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "  Neo4j Community $Neo4jVersion" -ForegroundColor Cyan
Write-Host "  Bolt  -> bolt://localhost:7687" -ForegroundColor Green
Write-Host "  Browser -> http://localhost:7474" -ForegroundColor Green
Write-Host "  Login: neo4j / password" -ForegroundColor Green
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "  Keep this window open while GraphRAG runs" -ForegroundColor Yellow
Write-Host ""

& "$Neo4jDir\bin\neo4j.bat" console
