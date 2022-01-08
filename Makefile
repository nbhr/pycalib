# vim: set noexpandtab:

all:
	@echo "make unittest"
	@echo "make clean"
	@echo "make build"
	@echo "make check"
	@echo "make upload-test"
	@echo "make upload-pypi"

unittest:
	python3 -m unittest discover tests

clean:
	rm -rf dist build pycalib.egg-info

build:
	python3 setup.py sdist
	python3 setup.py bdist_wheel

check:
	twine check dist/*

upload-test:
	twine upload --repository testpypi dist/*

upload-pypi:
	twine upload --repository pypi dist/*

