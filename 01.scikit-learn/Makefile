#!/usr/bin/make -f

IMG_NAME := train-sklearn
IMG_TAG := 1.0.0

.PHONY: build
build: ## build docker image
	docker build \
		--network host \
		-t ${IMG_NAME} -f Dockerfile .

.PHONY: run
run: ## run docker container
	docker run -it \
		--network=host \
		-p 8888:8888 \
		--rm --name ${IMG_NAME} ${IMG_NAME} || :

.PHONY: exec
exec: ## exec docker container
	docker exec -it ${IMG_NAME} /bin/bash

.PHONY: stop
stop: ## stop docker container
	docker stop ${IMG_NAME} || :

.PHONY: logs
logs: ## log docker container
	docker logs -f ${IMG_NAME} || :

.PHONY: clean
clean: ## stop docker container and remove docker image
	docker stop ${IMG_NAME} || :
	docker rm ${IMG_NAME} || :
	docker rmi ${IMG_NAME} ${IMG_NAME}:${IMG_TAG} || :

help: ## Display this help screen
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
