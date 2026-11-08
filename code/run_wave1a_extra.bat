@echo off
rem run_wave1a_extra.bat - second queue of SSL Wave 1a. Waits until the first queue writes "QUEUE DONE",
rem then: SimSiam, Barlow Twins (50 epochs each), supervised reference, final evaluation. One GPU job at a time.
rem Resumes like the first queue. Stop: create C:\SSL_Wave1\logs\wave1a.stop (also stops the first queue).
setlocal enabledelayedexpansion
set PYTHONUTF8=1
set PHYSICAL_AI_ROOT=C:\SSL_Wave1
cd /d C:\SSL_Wave1\code\ssl
set L=C:\SSL_Wave1\logs
set Q=%L%\wave1a_queue.log
echo !date! !time! EXTRA started, waiting for the first queue>> %Q%
:wait
if exist %L%\wave1a.stop goto :stopped
findstr /c:"QUEUE DONE" %Q% >nul 2>&1
if errorlevel 1 (
  timeout /t 300 /nobreak >nul
  goto :wait
)
for %%M in (simsiam barlow) do (
  if exist %L%\wave1a.stop goto :stopped
  echo !date! !time! START %%M>> %Q%
  python ssl_train_ext.py --method %%M --epochs 50 --workers 2 --gate-margin -1 --knn-every 10 2>> %L%\wave1a_%%M.err
  echo !date! !time! END %%M exit !errorlevel!>> %Q%
)
if exist %L%\wave1a.stop goto :stopped
echo !date! !time! START supervised>> %Q%
python sup_ref.py 2>> %L%\wave1a_supervised.err
echo !date! !time! END supervised exit !errorlevel!>> %Q%
echo !date! !time! START final_eval>> %Q%
python final_eval.py > %L%\wave1a_final_eval.out 2>&1
echo !date! !time! END final_eval exit !errorlevel!>> %Q%
echo !date! !time! EXTRA DONE>> %Q%
goto :end
:stopped
echo !date! !time! STOP file - extra queue ends>> %Q%
:end
endlocal
