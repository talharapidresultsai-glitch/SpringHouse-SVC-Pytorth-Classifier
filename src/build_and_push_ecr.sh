#!/usr/bin/env bash

# The argument to this script is the image name. This will be used as the image on the local
# machine.
#image=$1
image="springhouse-classifier"

if [ "$image" == "" ]
then
    echo "Usage: $0 <image-name>"
    exit 1
fi

chmod +x classifier/train
chmod +x classifier/serve

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi

# Define the region
region=${region:-us-east-2}


fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"

# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${image}" > /dev/null
fi

# Get the login command from ECR and execute it directly
aws ecr get-login-password --region "${region}" | docker login --username AWS --password-stdin "${account}".dkr.ecr."${region}".amazonaws.com

# Build the docker image locally
# with the full name.

docker build -t ${image} .
docker tag ${image} ${fullname}

docker push ${fullname}
