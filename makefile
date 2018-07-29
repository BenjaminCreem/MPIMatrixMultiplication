FCC = mpicc
LDC = mpicc
LD_FLAGS= -Wall -fopenmp 
FLAGS= -Wall -fopenmp -g
PROG = matMult 
RM = /bin/rm
OBJS = mm.o

#all rule
all: $(PROG)

$(PROG): $(OBJS)
	$(LDC) $(LD_FLAGS) $(OBJS) -o $(PROG)

%.o: %.c
	$(FCC) $(FLAGS) -c $<

#clean rule
clean:
	$(RM) -rf *.o $(PROG) *.mod
