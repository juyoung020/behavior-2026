# GPU 메모리(전용 = VRAM, 공유 = VRAM 이 모자랄 때 WDDM 이 대신 쓰는 시스템 RAM)를 2 초마다 CSV 로 남긴다 -- 검은 프레임 원인 가르기용.
#   powershell -ExecutionPolicy Bypass -File C:\behavior-2026\tools\gpu_mem_sampler.ps1 -Out <csv> -Seconds 600
# Windows 성능 카운터 \GPU Adapter Memory(*)\Dedicated Usage, Shared Usage (어댑터별 바이트). 끝나면 스스로 멈춘다.
param([Parameter(Mandatory = $true)][string]$Out, [int]$Seconds = 600, [int]$Every = 2)
"time,adapter,dedicated_mib,shared_mib" | Out-File -Encoding ascii $Out
$end = (Get-Date).AddSeconds($Seconds)
while ((Get-Date) -lt $end) {
    try {
        $s = Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage', '\GPU Adapter Memory(*)\Shared Usage' -ErrorAction Stop
        $t = Get-Date -Format HH:mm:ss
        $rows = @{}
        foreach ($c in $s.CounterSamples) {
            $inst = $c.InstanceName
            if (-not $rows.ContainsKey($inst)) { $rows[$inst] = @{ d = 0; s = 0 } }
            if ($c.Path -like '*dedicated usage') { $rows[$inst].d = $c.CookedValue } else { $rows[$inst].s = $c.CookedValue }
        }
        foreach ($k in $rows.Keys) {
            "$t,$k,$([math]::Round($rows[$k].d / 1MB)),$([math]::Round($rows[$k].s / 1MB))" | Out-File -Append -Encoding ascii $Out
        }
    } catch {}
    Start-Sleep -Seconds $Every
}
