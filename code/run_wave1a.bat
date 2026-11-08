@echo off
rem run_wave1a.bat - SSL Wave 1a queue: 4 methods x 50 epochs, one GPU job at a time.
rem Resumes automatically (each method continues from runs\ssl_<method>\last.pth; finished ones end at once).
rem Stop everything: create C:\SSL_Wave1\logs\wave1a.stop   (delete it before restarting)
setlocal enabledelayedexpansion
set PYTHONUTF8=1
set PHYSICAL_AI_ROOT=C:\SSL_Wave1
cd /d C:\SSL_Wave1\code\ssl
set Q=C:\SSL_Wave1\logs\wave1a_queue.log
for %%M in (simclr byol vicreg dino) do (
  if exist C:\SSL_Wave1\logs\wave1a.stop (
    echo %date% %time% STOP file - queue ends before %%M>> %Q%
    goto :end
  )
  echo %date% %time% START %%M>> %Q%
  python ssl_train.py --method %%M --epochs 50 --workers 2 --gate-margin -1 --knn-every 10 2>> C:\SSL_Wave1\logs\wave1a_%%M.err
  echo %date% %time% END %%M exit !errorlevel!>> %Q%
)
echo %date% %time% QUEUE DONE>> %Q%
:end
endlocal
