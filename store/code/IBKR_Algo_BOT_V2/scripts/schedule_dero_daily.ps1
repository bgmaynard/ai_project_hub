# DERO Scheduled Task Installer
# Run as Administrator to create Windows Task Scheduler entries
#
# Creates tasks:
# - AI_BOT_DERO_DAILY: Runs at 5:15 PM ET on trading days
# - AI_BOT_DERO_WEEKLY: Runs at 6:00 PM ET on Fridays

param(
    [switch]$Remove,
    [switch]$Daily,
    [switch]$Weekly,
    [switch]$All
)

$ProjectRoot = "C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2"
$PythonPath = "$ProjectRoot\venv311\Scripts\python.exe"
$LogFile = "$ProjectRoot\logs\dero_scheduler.log"

# Ensure log directory exists
New-Item -ItemType Directory -Path "$ProjectRoot\logs" -Force | Out-Null

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "$timestamp | SCHEDULER | $Message"
    Write-Host $logMessage
    Add-Content -Path $LogFile -Value $logMessage
}

function Create-DeroDaily {
    $TaskName = "AI_BOT_DERO_DAILY"
    $Description = "DERO Daily Evaluation Report - Runs at 5:15 PM ET on trading days"
    $Script = "$ProjectRoot\scripts\run_dero_daily.py"

    Write-Log "Creating task: $TaskName"

    # Check if task exists
    $existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existingTask) {
        Write-Log "Task already exists, removing old task..."
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }

    # Create action
    $Action = New-ScheduledTaskAction -Execute $PythonPath -Argument "`"$Script`"" -WorkingDirectory $ProjectRoot

    # Create trigger (5:15 PM ET, Monday-Friday)
    $Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "17:15"

    # Create settings
    $Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd -AllowStartIfOnBatteries

    # Register task
    try {
        Register-ScheduledTask -TaskName $TaskName -Description $Description -Action $Action -Trigger $Trigger -Settings $Settings -RunLevel Highest
        Write-Log "Task $TaskName created successfully"
        return $true
    } catch {
        Write-Log "Failed to create task: $_"
        return $false
    }
}

function Create-DeroWeekly {
    $TaskName = "AI_BOT_DERO_WEEKLY"
    $Description = "DERO Weekly Evaluation Report - Runs at 6:00 PM ET on Fridays"
    $Script = "$ProjectRoot\scripts\run_dero_weekly.py"

    Write-Log "Creating task: $TaskName"

    # Check if task exists
    $existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existingTask) {
        Write-Log "Task already exists, removing old task..."
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }

    # Create action
    $Action = New-ScheduledTaskAction -Execute $PythonPath -Argument "`"$Script`"" -WorkingDirectory $ProjectRoot

    # Create trigger (6:00 PM ET on Fridays)
    $Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Friday -At "18:00"

    # Create settings
    $Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd -AllowStartIfOnBatteries

    # Register task
    try {
        Register-ScheduledTask -TaskName $TaskName -Description $Description -Action $Action -Trigger $Trigger -Settings $Settings -RunLevel Highest
        Write-Log "Task $TaskName created successfully"
        return $true
    } catch {
        Write-Log "Failed to create task: $_"
        return $false
    }
}

function Remove-DeroTasks {
    $tasks = @("AI_BOT_DERO_DAILY", "AI_BOT_DERO_WEEKLY")

    foreach ($taskName in $tasks) {
        $task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
        if ($task) {
            Write-Log "Removing task: $taskName"
            Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
            Write-Log "Task $taskName removed"
        } else {
            Write-Log "Task $taskName not found"
        }
    }
}

# Main execution
Write-Log "=========================================="
Write-Log "DERO Task Scheduler Configuration"
Write-Log "=========================================="

if ($Remove) {
    Remove-DeroTasks
    exit 0
}

if ($Daily -or $All) {
    Create-DeroDaily
}

if ($Weekly -or $All) {
    Create-DeroWeekly
}

if (-not $Daily -and -not $Weekly -and -not $All) {
    # Default: create both
    Create-DeroDaily
    Create-DeroWeekly
}

Write-Log "=========================================="
Write-Log "Scheduled tasks configured"
Write-Log ""
Write-Log "To verify tasks, run:"
Write-Log "  Get-ScheduledTask -TaskName 'AI_BOT_DERO_*'"
Write-Log ""
Write-Log "To run manually:"
Write-Log "  python scripts/run_dero_daily.py"
Write-Log "  python scripts/run_dero_weekly.py"
Write-Log "=========================================="
