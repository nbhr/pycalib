# vim: set noexpandtab:

all:
	@echo "make unittest"
	@echo "make clean"
	@echo "make build"
	@echo "make check"
	@echo "make upload-test"
	@echo "make upload-pypi"
	@echo
	@echo "make requirements.txt"

unittest:
	python3 -m unittest discover tests
	for i in ipynb/*.ipynb; do j=`basename $$i .ipynb`; grep -q test_$${j} tests/test_ipynb.py || echo "$$i is not tested"; done
	for i in ipynb/*.ipynb; do j=`basename $$i`; grep -q $${j} README.md || echo "$$i is not mentioned in README"; done

clean:
	rm -rf dist build pycalib.egg-info
	find -name '*~' -exec rm -vf {} +
	rm -vf ipynb/output*

build:
	python3 setup.py sdist
	python3 setup.py bdist_wheel

check:
	twine check dist/*

upload-test:
	twine upload --repository testpypi dist/*

upload-pypi:
	twine upload --repository pypi dist/*

requirements:
	pipreqs --use-local --force .

