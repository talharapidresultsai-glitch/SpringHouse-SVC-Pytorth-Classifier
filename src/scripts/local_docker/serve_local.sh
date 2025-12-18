#!/bin/sh

#image=$1
image="local/classifier"

docker run -v $(pwd)/test_dir:/opt/ml -p 8080:8080 -e LOG_URL='http://10.0.0.151:6000/log' --rm ${image} serve
