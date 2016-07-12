# Value function iteration on the GPU in python using pycuda

## Setup

This requires gcc version 4.9.x and above. See [here](https://gist.github.com/jtilly/2827af06e331e8e6b53c)
for instructions on compiling gcc locally on tesla.

The package `pycuda` has one dependency, which is out of date
on tesla and must be installed locally. The following installs
both packages locally:

```
# upgrade numpy for local user
pip2 install --user --upgrade numpy

# install pycuda
pip2 install --user pycuda
```

`pip2` installs packages for version 2.6, `pip` may also suffice.
Just don't install the package for version 3, you can check the
version with
```
pip --version
```

## Run the example

Run the example with
```
python vf.py
```
