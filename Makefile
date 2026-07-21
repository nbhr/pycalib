# vim: set noexpandtab:

VERSION := $(shell d=`date +%Y.%m.%d`; n=`git tag -l "v$$d.*" | wc -l`; echo $$d.$$((n+1)))
INIT    := pycalib/__init__.py

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

version:
	@echo $(VERSION)

bump:
	sed -i.bak -E "s/^__version__ *= *['\"].*['\"]/__version__ = '$(VERSION)'/" $(INIT)
	@rm -f $(INIT).bak
	@grep -n '__version__' $(INIT)

release: check-clean bump
	git add $(INIT)
	git commit -m "Release $(VERSION)"
	git tag -a v$(VERSION) -m "Release $(VERSION)"
	git push origin master
	git push origin v$(VERSION)
	gh release create v$(VERSION) --title "$(VERSION)" --generate-notes

check-clean:
	@git diff --quiet HEAD -- || \
		{ echo "tracked files are modified"; \
		  git status --short --untracked-files=no; exit 1; }

release-test: check-clean bump
	git add $(INIT)
	@git diff --cached --quiet || git commit -m "Test release $(VERSION)"
	git tag -a v$(VERSION) -m "Test release $(VERSION)"
	git push origin master v$(VERSION)
	gh release create v$(VERSION) --prerelease --title "$(VERSION)" --generate-notes
