docker build -t myimage .
docker run -d --name mycontainer -p 8000:8000 myimage
