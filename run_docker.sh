docker build -f docker/Dockerfile -t test/app .
docker run test/app -v models:/exp/models