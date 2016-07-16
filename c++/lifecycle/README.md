# Life-cycle model on the GPU using c++

## Setup

This requires gcc version 4.9.x and above. See [here](https://gist.github.com/jtilly/2827af06e331e8e6b53c)
for instructions on compiling gcc locally on tesla.

You also need to install the [boost](http://www.boost.org/doc/libs/1_61_0/more/getting_started/unix-variants.html#get-boost)
libraries, as well as the [NLopt](http://ab-initio.mit.edu/wiki/index.php/NLopt_Installation) optimization library:
```
# install boost header libraries, assuming you've compiled gcc 4.9.3 locally into
# the folder gcc-4.9.3
wget https://sourceforge.net/projects/boost/files/boost/1.61.0/boost_1_61_0.tar.gz
tar -xf boost_1_61_0.tar.gz
mv boost_1_61_0/include ~/gcc-4.9.3/

# install NLopt locally
wget http://ab-initio.mit.edu/nlopt/nlopt-2.4.2.tar.gz
tar -xf nlopt-2.4.2.tar.gz
cd nlopt-2.4.2
./configure --prefix=$HOME/gcc-4.9.3
make 
make install
cd ..

# clean up
rm nlopt-2.4.2.tar.gz
rm boost_1_61_0.tar.gz
rm -rf boost_1_61_0
rm -rf nlopt-2.4.2
```

## Run the example

Compile and run the example with
```
make
./main
```
