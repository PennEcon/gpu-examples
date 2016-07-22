# Maximum Likelihood estimation on the GPU using c++
## Warning
This version is *extremely* preliminary. It is very likely that there are typos in the code. However, the optimization routine is working properly and I will 
leave it as it is for the moment. 
## Setup
This requires gcc version 4.9.x and above. See [here](https://gist.github.com/jtilly/2827af06e331e8e6b53c)
for instructions on compiling gcc locally on tesla.

You also need to install the [NLopt](http://ab-initio.mit.edu/wiki/index.php/NLopt_Installation) optimization library.

See [here](https://gist.github.com/njanetos/1c73f621367c2bed88e9be96445d5460) for instructions on installing
NLopt locally on Tesla.

## Run the example

Compile and run the example with
```
make parallel
./CudaT
```
The code will generate a csv file called PARAMETERSFOUND.csv with the ML estimates.  The output of the file compiled will be stored in nohupPARALLR.out
