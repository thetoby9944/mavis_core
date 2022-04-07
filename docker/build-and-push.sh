#!/bin/bash
IMAGE_NAME=mavis
#VERSION="prod-0.1.${BITBUCKET_BUILD_NUMBER}"
VERSION=latest

read -p 'Docker ID: ' DOCKERHUB_USERNAME
read -p 'Password: ' DOCKERHUB_PASSWORD

echo ${DOCKERHUB_PASSWORD} | docker login --username "$DOCKERHUB_USERNAME" --password-stdin
docker build . --file Dockerfile --tag ${IMAGE_NAME}
IMAGE=${DOCKERHUB_USERNAME}/${IMAGE_NAME}
docker tag "${IMAGE_NAME}" "${IMAGE}:${VERSION}"
docker push "${IMAGE}:${VERSION}"
