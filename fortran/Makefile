# Set Fortran compiler
FORT_COMPILER = gfortran

# Main program source file
MAIN_SRC = main.f90

# Object files
OBJ = main.o

# Executable name
EXEC = main

# Compile rules
$(EXEC): $(OBJ)
	$(FORT_COMPILER) -o $@ $^

%.o: %.f90
	$(FORT_COMPILER) -c -O3 $<

clean:
	rm -f $(EXEC) $(OBJ)