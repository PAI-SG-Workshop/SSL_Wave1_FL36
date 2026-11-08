# ssl_env_check.ps1 - READ-ONLY environment check for SSL Wave 1a (GTX 1060 laptop)
# Run from the same conda cmd window:
#   powershell -ExecutionPolicy Bypass -File ssl_env_check.ps1
# Writes Desktop\ssl_env_report.txt. Changes nothing on the system.

$ErrorActionPreference = "Continue"
$out = Join-Path ([Environment]::GetFolderPath("Desktop")) "ssl_env_report.txt"
if (Test-Path $out) { Remove-Item $out }
function W($s) { $s | Tee-Object -FilePath $out -Append }

W ("=== ssl_env_check " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " ===")

W "--- OS / CPU / RAM"
$os = Get-CimInstance Win32_OperatingSystem
W ("OS: {0} build {1}" -f $os.Caption, $os.BuildNumber)
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
W ("CPU: {0}, cores {1}, threads {2}" -f $cpu.Name.Trim(), $cpu.NumberOfCores, $cpu.NumberOfLogicalProcessors)
W ("RAM: {0:N1} GB total, {1:N1} GB free" -f ($os.TotalVisibleMemorySize/1MB), ($os.FreePhysicalMemory/1MB))
$lastBoot = $os.LastBootUpTime
W ("Last boot: {0}" -f $lastBoot)

W "--- Disks (fixed)"
Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" | ForEach-Object {
  W ("{0} free {1:N0} GB of {2:N0} GB" -f $_.DeviceID, ($_.FreeSpace/1GB), ($_.Size/1GB))
}

W "--- GPU"
W (nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu,power.draw,clocks.sm,clocks.max.sm --format=csv 2>&1 | Out-String)
W (nvidia-smi -q -d PERFORMANCE 2>&1 | Out-String)

W "--- Power plan (0x00000000 = never)"
W (powercfg /getactivescheme 2>&1 | Out-String)
W "Sleep after (AC/DC):"
W (powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE 2>&1 | Select-String "Current" | Out-String)
W "Hibernate after (AC/DC):"
W (powercfg /query SCHEME_CURRENT SUB_SLEEP HIBERNATEIDLE 2>&1 | Select-String "Current" | Out-String)
W "Lid close action (0 = do nothing, 1 = sleep):"
W (powercfg /query SCHEME_CURRENT SUB_BUTTONS LIDACTION 2>&1 | Select-String "Current" | Out-String)
$bat = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue
if ($bat) { W ("Battery: {0}% , status {1} (2 = on AC)" -f $bat.EstimatedChargeRemaining, $bat.BatteryStatus) }

W "--- Tools"
W ("git: " + (git --version 2>&1 | Out-String).Trim())
W ("gh: " + (gh --version 2>&1 | Select-Object -First 1 | Out-String).Trim())
W ("conda: " + (conda --version 2>&1 | Out-String).Trim())
W "conda envs:"
W (conda env list 2>&1 | Out-String)

W "--- Network (HTTPS reachability)"
foreach ($h in @("github.com", "pypi.org", "download.pytorch.org")) {
  $ok = Test-NetConnection $h -Port 443 -InformationLevel Quiet -WarningAction SilentlyContinue
  W ("{0}: {1}" -f $h, $ok)
}

W "--- Python / torch in the CURRENT env (read-only)"
$py = @'
import sys, time
print("python", sys.version.split()[0], sys.executable)
try:
    import torch
    print("torch", torch.__version__, "| cuda build", torch.version.cuda)
    print("cuda available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0))
        print("arch list", torch.cuda.get_arch_list())
        for dtype in (torch.float32, torch.float16):
            a = torch.randn(2048, 2048, device="cuda", dtype=dtype)
            b = torch.randn(2048, 2048, device="cuda", dtype=dtype)
            for _ in range(3):
                a @ b
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(20):
                a @ b
            torch.cuda.synchronize(); sec = (time.time() - t0) / 20
            print("matmul 2048 %s: %.2f TFLOPS" % (dtype, 2 * 2048 ** 3 / sec / 1e12))
except Exception as e:
    print("torch check failed:", repr(e))
for m in ("torchvision", "numpy", "cv2", "PIL", "scipy", "sklearn", "tqdm"):
    try:
        mod = __import__(m)
        print(m, getattr(mod, "__version__", "?"))
    except Exception:
        print(m, "MISSING")
'@
$tmp = Join-Path $env:TEMP "ssl_env_check_tmp.py"
Set-Content -Path $tmp -Value $py -Encoding ASCII
W (python $tmp 2>&1 | Out-String)
Remove-Item $tmp -ErrorAction SilentlyContinue

W "--- GPU after torch test"
W (nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,memory.used --format=csv 2>&1 | Out-String)

W ("=== done. Report: " + $out + " ===")
