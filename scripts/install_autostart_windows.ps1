# =====================================================================
# Market Sentinel AI — Démarrage automatique avec Windows
# =====================================================================
# Ce script crée une tâche planifiée qui lance le service d'arrière-plan
# à l'ouverture de session Windows, en mode silencieux (pythonw, sans
# fenêtre console).
#
# Utilisation (PowerShell) :
#   .\scripts\install_autostart_windows.ps1            # installe
#   .\scripts\install_autostart_windows.ps1 -Remove    # désinstalle
#
# Astuce : exécuter une première fois manuellement
#   python scripts\run_service.py
# pour vérifier que tout fonctionne avant d'activer le démarrage auto.
# =====================================================================

param(
    [switch]$Remove
)

$TaskName    = "MarketSentinelAI"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ServiceScript = Join-Path $ProjectRoot "scripts\run_service.py"

if ($Remove) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Tâche planifiée '$TaskName' supprimée." -ForegroundColor Yellow
    return
}

# Recherche de pythonw.exe (exécution sans fenêtre console).
$pythonw = (Get-Command pythonw.exe -ErrorAction SilentlyContinue).Source
if (-not $pythonw) {
    Write-Host "pythonw.exe introuvable. Installez Python (python.org) et cochez 'Add to PATH'." -ForegroundColor Red
    return
}

$Action  = New-ScheduledTaskAction -Execute $pythonw -Argument "`"$ServiceScript`"" -WorkingDirectory $ProjectRoot
$Trigger = New-ScheduledTaskTrigger -AtLogOn
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description "Service d'analyse de marché Market Sentinel AI" -Force | Out-Null

Write-Host "Tâche planifiée '$TaskName' installée." -ForegroundColor Green
Write-Host "Elle démarrera automatiquement à la prochaine ouverture de session."
Write-Host "Pour la lancer maintenant : Start-ScheduledTask -TaskName $TaskName"
