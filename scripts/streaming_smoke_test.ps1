Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ====== Config (override via env in GitHub Actions if you want) ======
if (-not $env:KAFKA_BOOTSTRAP_SERVERS) { $env:KAFKA_BOOTSTRAP_SERVERS = "localhost:9092" }
if (-not $env:KAFKA_TOPIC_IN)          { $env:KAFKA_TOPIC_IN          = "vitals.in" }
if (-not $env:KAFKA_TOPIC_DLQ)         { $env:KAFKA_TOPIC_DLQ         = "vitals.dlq" }

# CI-friendly unique group id
$runId      = if ($env:GITHUB_RUN_ID)      { $env:GITHUB_RUN_ID }      else { "local" }
$runAttempt = if ($env:GITHUB_RUN_ATTEMPT) { $env:GITHUB_RUN_ATTEMPT } else { "0" }
$env:KAFKA_GROUP_ID = "ci-consumer-$runId-$runAttempt"

$env:IDEMPOTENCY_DB     = "/tmp/seen_events_ci.db"
$env:VITALS_AUDIT_PATH  = "/tmp/vitals_audit.log"
$consumerLog            = "/tmp/consumer.log"

# Compose file
$compose = "docker-compose.kafka.yml"

function Write-Section($title) {
  Write-Host ""
  Write-Host "==== $title ====" -ForegroundColor Cyan
}

function Safe-Run([scriptblock]$cmd) {
  try { & $cmd } catch { Write-Host "(ignored) $($_.Exception.Message)" -ForegroundColor DarkYellow }
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

  # NOTE: choose brokers consistent with your listener mapping
  $brokers = $env:KAFKA_BOOTSTRAP_SERVERS

  Write-Section "consume head: vitals.in"
  Safe-Run { docker compose -f $compose exec -T redpanda rpk topic consume $env:KAFKA_TOPIC_IN --brokers $brokers -n 5 }

  Write-Section "consume head: DLQ"
  Safe-Run { docker compose -f $compose exec -T redpanda rpk topic consume $env:KAFKA_TOPIC_DLQ --brokers $brokers -n 10 }

  exit 1
}

# ====== Clean files ======
Safe-Run { Remove-Item -Force $env:VITALS_AUDIT_PATH -ErrorAction SilentlyContinue }
Safe-Run { Remove-Item -Force $env:IDEMPOTENCY_DB    -ErrorAction SilentlyContinue }
Safe-Run { Remove-Item -Force $consumerLog           -ErrorAction SilentlyContinue }

# ====== Start Redpanda ======
Write-Section "docker compose up"
docker compose -f $compose up -d

# ====== Wait for broker readiness ======
Write-Section "wait for redpanda readiness (rpk cluster info)"
$maxSeconds = 60
$ready = $false
for ($i=0; $i -lt $maxSeconds; $i++) {
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

# ====== Connectivity probe (host-level) ======
Write-Section "host connectivity probe"
$parts = $env:KAFKA_BOOTSTRAP_SERVERS.Split(":")
if ($parts.Count -ne 2) { Fail-Smoke "invalid KAFKA_BOOTSTRAP_SERVERS=$($env:KAFKA_BOOTSTRAP_SERVERS)" }
$host = $parts[0]; $port = [int]$parts[1]
$tnc = Test-NetConnection -ComputerName $host -Port $port -WarningAction SilentlyContinue
$tnc | Select-Object ComputerName,RemotePort,TcpTestSucceeded | Format-Table -AutoSize
if (-not $tnc.TcpTestSucceeded) { Fail-Smoke "cannot reach broker at $($env:KAFKA_BOOTSTRAP_SERVERS)" }

# ====== Run your producer + consumer (YOU plug in your real commands) ======
# Example placeholders — replace with your repo's actual commands:
#   - producer: emits N messages to vitals.in
#   - consumer: reads and writes to $env:VITALS_AUDIT_PATH
Write-Section "run producer"
try {
  # EXAMPLE: python -m src.streaming.producer --n 20
  python -m src.streaming.producer --n 20
} catch {
  Fail-Smoke "producer failed: $($_.Exception.Message)"
}

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
    $len = (Get-Item $env:VITALS_AUDIT_PATH).Length
    if ($len -gt 0) { break }
  }
} while ((Get-Date) -lt $deadline)

# Stop consumer if still running (best effort)
Safe-Run {
  if (-not $consumerProc.HasExited) {
    $consumerProc.CloseMainWindow() | Out-Null
    Start-Sleep -Seconds 1
    if (-not $consumerProc.HasExited) { $consumerProc.Kill() }
  }
}

# ====== Assert success ======
if (-not (Test-Path $env:VITALS_AUDIT_PATH)) { Fail-Smoke "audit file was never created" }
if ((Get-Item $env:VITALS_AUDIT_PATH).Length -le 0) { Fail-Smoke "audit file is empty (consumer did not process any events)" }

Write-Section "SUCCESS"
Write-Host "✅ streaming smoke test passed"
Get-Content $env:VITALS_AUDIT_PATH -Tail 20
