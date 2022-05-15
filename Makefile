export ROOT_DIR=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
export EXCLUDE_FILES=$(shell cat ${ROOT_DIR}/.gitignore | xargs -tI{} echo '--exclude {}' | xargs)


upload:
	rsync -av ${EXCLUDE_FILES} --exclude .git ${ROOT_DIR} ${REMOTE_MACHINE}
lint:
	black .
	isort .
