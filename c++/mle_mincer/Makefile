
#---------------------
#Housekeeping        #
#---------------------

#1. C++ compiler
CC=nvcc

#2. Flags
CFLAGS= 

#3. Name of output
OUTPUTPARALLEL=CudaT
#4. Files to run in main:
SOURCEPARALLEL=CUDAMLE.cu

#5. Home directory. in TESLA is /home/razu/
HOME=/home/razu/

#6. Dynamic libraries 
DLIB=-lgsl -lgslcblas

#7. Set number of threads	
#export OMP_NUM_THREADS=30

#-------------------------------------
#3. Include directories:
INCDIR1=/usr/include/c++/3.4.6/backward/
INCDIR2=/usr/lib64/R/library/BH/include/
INCDIR3=$(HOME)install/include/

#Generating the INC for all the paths
INC=$(INCDIR1) $(INCDIR2) $(INCDIR3) 
#-> This is for tesla
#INC=$(INCDIR3)
INC_PARAMS=$(foreach d, $(INC), -I$d)
#-------------------------------------

#4. Lnlopt library 
LNLOPT=-L$(HOME)install/lib/ -lnlopt




nloptinst:
	wget http://ab-initio.mit.edu/nlopt/nlopt-2.4.2.tar.gz -P $(HOME)
	tar  -zxvf $(HOME)nlopt-2.4.2.tar.gz -C$(HOME)
	mkdir $(HOME)/install
	cd $(HOME)nlopt-2.4.2 && ./configure --prefix=$(HOME)install && make && make install

parallel:
	$(CC) $(SOURCEPARALLEL) $(INC_PARAMS) -L$(HOME)install/lib/ -lnlopt   $(DLIB) -o $(OUTPUTPARALLEL)
	

parallelrun: parallel
	nohup ./$(OUTPUTPARALLEL)  > nohupPARALLR.out&

nloptuninst:
	rm $(HOME)nlopt-2.4.2.tar.gz
	rm -r $(HOME)nlopt-2.4.2




clean: 
	rm $(OUTPUT)



