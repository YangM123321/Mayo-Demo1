Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ====== Config ======
if (-not $env:KAFKA_BOOTSTRAP_SERVERS) { $env:KAFKA_BOOTSTRAP_SERVERS = "localhost:9092" }
if (-not $env:KAFKA_TOPIC_IN)          { $env:KAFKA_TOPIC_IN          = "vitals.in" }
if (-not $env:KAFKA_TOPIC_DLQ)         { $env:KAFKA_TOPIC_DLQ         = "vitals.dlq" }

$runId      = if ($env:GITHUB_RUN_ID)      { $env:GITHUB_RUN_ID }      else { "local" }
$runAttempt = if ($env:GITHUB_RUN_ATTEMPT) { $env:GITHUB_RUN_ATTEMPT } else { "0" }
$env:KAFKA_GROUP_ID = "ci-consumer-$runId-$runAttempt"

$env:IDEMPOTENCY_DB    = "/tmp/seen_events_ci.db"
$env:VITALS_AUDIT_PATH = "/tmp/vitals_audit.log"
$consumerLog           = "/tmp/consumer.log"

$compose = "docker-compose.kafka.yml"

# IMPORTANT:
# - From the GitHub runner host: localhost:9092 is correct IF compose publishes 9092.
# - From INSIDE the redpanda container: use redpanda:9092
$brokersInContainer = "redpanda:9092"
$brokersOnHost      = $env:KAFKA_BOOTSTRAP_SERVERS

function Write-Section([string]$title) {
  Write-Host ""
  Write-Host "==== $title ====" -ForegroundColor Cyan
}

function Safe-Run([scriptblock]$cmd) {
  try { & $cmd } catch { Write-Host "(ignored) $($_.Exception.Message)" -ForegroundColor DarkYellow }
}

function Wait-ForPort([string]$host, [int]$port, [int]$timeoutSeconds) {
  $deadline = (Get-Date).AddSeconds($timeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    try {
      $client = [System.Net.Sockets.TcpClient]::new()
      $iar = $client.BeginConnect($host, $port, $null, $null)
      if ($iar.AsyncWaitHandle.WaitOne(500)) {
        $client.EndConnect($iar) | Out-Null
        $client.Close()
        return $true
      }
      $client.Close()
    } catch {
      # ignore and retry
    }
    Start-Sleep -Milliseconds 500
  }
  return $false
}

function Fail-Smoke {
  param([string]$Message = "streaming smoke test failed")

  Write-Host "❌ $Message" -ForegroundColor Red

  Write-Section "consumer log (tail)"
  Safe-Run { Get-Content $consumerLog -Tail 200 }

  Write-Section "redpanda logs"
  Safe-Run { docker compose -f $compose logs --no-color redpanda }

  Write-Section "audit file"
  Safe-Run { Get-Item $env:VITALS_AUDIT_PATH | Format-List Name,Length,LastWriteTime }
  Safe-Run { Get-Content $env:VITALS_AUDIT_PATH -Tail 50 }

  Write-Section "topics"
  Safe-Run { docker compose -f $compose exec -T redpanda rpk topic list }

  Write-Section "describe topics"
  Safe-Run { docker compose -f $compose exec -T redpanda rpk topic describe $env:KAFKA_TOPIC_IN }
  Safe-Run { docker compose -f $compose exec -T redpanda rpk topic describe $env:KAFKA_TOPIC_DLQ }

  Write-Section "consume head: vitals.in (from inside container)"
  Safe-Run { docker compose -f $compose exec -T redpanda rpk topic consume $env:KAFKA_TOPIC_IN --brokers $brokersInContainer -n 5 }

  Write-Section "consume head: DLQ (from inside container)"
  Safe-Run { docker compose -f $compose exec -T redpanda rpk topic consume $env:KAFKA_TOPIC_DLQ --brokers $brokersInContainer -n 10 }

  exit 1
}

# ====== Clean files ======
Safe-Run { Remove-Item -Force $env:VITALS_AUDIT_PATH -ErrorAction SilentlyContinue }
Safe-Run { Remove-Item -Force $env:IDEMPOTENCY_DB    -ErrorAction SilentlyContinue }
Safe-Run { Remove-Item -Force $consumerLog           -ErrorAction SilentlyContinue }

# ====== Start Redpanda ======
Write-Section "docker compose up"
docker compose -f $compose up -d

# ====== Wait for broker readiness (rpk works inside container) ======
Write-Section "wait for redpanda readiness (rpk cluster info)"
$ready = $false
for ($i=0; $i -lt 60; $i++) {
  try {
    docker compose -f $compose exec -T redpanda rpk cluster info | Out-Null
    $ready = $true
    break
  } catch {
    Start-Sleep -Seconds 1
  }
}
if (-not $ready) { Fail-Smoke "redpanda never became ready" }

# ====== Ensure topics exist ======
Write-Section "ensure topics exist"
Safe-Run { docker compose -f $compose exec -T redpanda rpk topic create $env:KAFKA_TOPIC_IN  --partitions 1 --replicas 1 }
Safe-Run { docker compose -f $compose exec -T redpanda rpk topic create $env:KAFKA_TOPIC_DLQ --partitions 1 --replicas 1 }
Safe-Run { docker compose -f $compose exec -T redpanda rpk topic list }

# ====== Host connectivity probe (Linux-safe) ======
# ====== Host connectivity probe (Linux-safe) ======
Write-Section "host connectivity probe"
$parts = $brokersOnHost.Split(":")
if ($parts.Count -ne 2) { Fail-Smoke "invalid KAFKA_BOOTSTRAP_SERVERS=$brokersOnHost" }

$brokerHost = $parts[0]
$brokerPort = [int]$parts[1]

Write-Host "Checking TCP connect to ${brokerHost}:${brokerPort} ..."
if (-not (Wait-ForPort -host $brokerHost -port $brokerPort -timeoutSeconds 20)) {
  Fail-Smoke "cannot reach broker at $brokersOnHost (is port 9092 published in docker-compose?)"
}
Write-Host "✅ TCP reachable"


# ====== Run producer ======
Write-Section "run producer"
try {
  python -m src.streaming.producer --n 20
} catch {
  Fail-Smoke "producer failed: $($_.Exception.Message)"
}

# ====== Run consumer (background) ======
Write-Section "run consumer (background)"
$consumerProc = Start-Process -FilePath "python" `
  -ArgumentList @("-m","src.streaming.consumer","--max-messages","20") `
  -NoNewWindow -PassThru `
  -RedirectStandardOutput $consumerLog -RedirectStandardError $consumerLog

# ====== Wait for audit file to have content ======
Write-Section "wait for audit output"
$deadline = (Get-Date).AddSeconds(45)
do {
  Start-Sleep -Milliseconds 500
  if (Test-Path $env:VITALS_AUDIT_PATH) {
    if ((Get-Item $env:VITALS_AUDIT_PATH).Length -gt 0) { break }
  }
} while ((Get-Date) -lt $deadline)

# Stop consumer (best effort)
Safe-Run {
  if (-not $consumerProc.HasExited) {
    try { $consumerProc.Kill() } catch { }
  }
}

# ====== Assert success ======
if (-not (Test-Path $env:VITALS_AUDIT_PATH)) { Fail-Smoke "audit file was never created" }
if ((Get-Item $env:VITALS_AUDIT_PATH).Length -le 0) { Fail-Smoke "audit file is empty (consumer did not process any events)" }

Write-Section "SUCCESS"
Write-Host "✅ streaming smoke test passed"
Get-Content $env:VITALS_AUDIT_PATH -Tail 20
