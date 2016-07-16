#---------------------
#Housekeeping        #
#---------------------

#1. C++ compiler
CC=nvcc

#2. Name of output
OUTPUT=main

#3. Files to run in main:
SOURCE=main.cu



myprog:
	$(CC) $(SOURCE) -o $(OUTPUT)

execute: myprog
	./$(OUTPUT)


