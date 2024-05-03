#!/bin/bash

if [[ "$OSTYPE" == "darwin"* ]]; then
	sed -i '' 's/import sklearn.cluster as cluster/#import sklearn.cluster as cluster/' graphlearning/clustering.py
else
	sed -i 's/import sklearn.cluster as cluster/#import sklearn.cluster as cluster/' graphlearning/clustering.py
fi

pdoc3 --template-dir ./pdoc/templates --html --force -o docs/ ./graphlearning > /dev/null

if [[ "$OSTYPE" == "darwin"* ]]; then
	sed -i '' 's/#import sklearn.cluster as cluster/import sklearn.cluster as cluster/' graphlearning/clustering.py
else
	sed -i 's/#import sklearn.cluster as cluster/import sklearn.cluster as cluster/' graphlearning/clustering.py
fi

mv docs/graphlearning/* docs/
rmdir docs/graphlearning/
ls -1 docs/ | sed 's/^/docs\//'

#Also, it is possible to omit functions by using the module dictionary within pdoc. An easy way to do it is to put the following at the top of each .py file in which you want to omit certain functions:

#__pdoc__ = {}
#__pdoc__['function_name'] = False

#If you want to omit class methods instead, simply put 

#__pdoc__ = {}
#__pdoc__['class_name.method_name'] = False
#
#within the class definition.

