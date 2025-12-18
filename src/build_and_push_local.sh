#!/usr/bin/env bash

# The argument to this script is the image name. This will be used as the image on the local
# machine.
#image=$1
image="local_classifier"

if [ "$image" == "" ]
then
    echo "Usage: $0 <image-name>"
    exit 1
fi

chmod +x classifier/train
chmod +x classifier/serve

fullname="local/${image}:latest"

# Build the docker image locally
# with the full name.

#--no-cache
#docker build --no-cache -t ${image} .
docker build -t ${image} .
docker tag ${image} ${fullname}

#docker push ${fullname}
