#!/usr/bin/bash
docker build . -t localtest:latest -f docker/Dockerfile
docker run --rm -p 5000:5000 -v /$(pwd)/models:/exp/models localtest:latest