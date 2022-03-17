all:
	python3 setup.py install --user --prefix=
	bash generate_docs.sh
