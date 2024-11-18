install:
	pip install . --verbose
	rm -rf *.egg-info
	rm -rf build
uninstall:
	pip uninstall --yes pykatsevich
clean:
	rm -rf *.egg-info
	rm -rf build