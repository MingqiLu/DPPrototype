#/bin/sh

echo 'Complie the C++ utility library'
python setup.py build_ext -b ./cutils/ --swig-opts=-c++
