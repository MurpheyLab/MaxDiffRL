#!/bin/bash

container=test
image=maxdiff
port=6080
url="http://localhost:$port"

cleanup() {
	docker stop $container >/dev/null
	docker rm $container >/dev/null
}

image_found=$(docker images -q ${image})
if [ -z "$image_found" ]; then
	echo "Building image..."
	docker build -t $image --label latest .
	echo " "
fi


running=$(docker ps -a -q --filter "name=${container}")
if [ -n "$running" ]; then
	echo "Stopping and removing the previous session..."
	cleanup
fi

echo "Setting up the graphical application container..."
echo "Point your web browser to ${url}"
echo "When ready to end docker session, close terminal"

docker run \
  -d \
  --name $container \
  -v docker_test_results:/home/user/work/results \
  -p $port:6080 \
  --env "APP=xterm" \
  $image >/dev/null

print_app_output() {
	docker cp $container:/var/log/supervisor/graphical-app-launcher.log - \
		| tar xO
	result=$(docker cp $container:/tmp/graphical-app.return_code - \
		| tar xO)
	cleanup
}

trap "docker stop $container >/dev/null && print_app_output" SIGINT SIGTERM

docker wait $container >/dev/null

print_app_output
echo "Shutdown complete"