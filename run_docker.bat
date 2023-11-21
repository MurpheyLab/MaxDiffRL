@echo off
setlocal EnableDelayedExpansion

SET container=test
SET image=maxdiff
SET port=6080
SET url=http://localhost:%port%

echo Checking for docker image...
docker images -q %image% |findstr .  > nul 2>&1 && set placeholder=blah || ( echo Building image... & docker build -t %image% --label latest . & set placeholder=blah )

echo Checking for docker container...
docker ps -a -q --filter name=%container%  > nul 2>&1 && (echo Stopping and removing the previous session...  & docker stop %container%  > nul 2>&1 & docker rm %container%  > nul 2>&1) || set placeholder=blah

echo Setting up the graphical application container...
echo:
echo Point your web browser to %url%
echo When ready to end docker session, close terminal

docker run -d --name %container% -v docker_test_results:/home/user/work/results -p %port%:6080 --env "APP=xterm" %image%  > nul 2>&1