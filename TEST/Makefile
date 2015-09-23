# include the environment, compilation, and linking flags

include ../Make_flags
include ../Link_flags

#------------------------------------------------

REAL= double

SOBJS= driver.o COMMON/shared_utils.o
INCLUDE += -I./COMMON

USE_MPI=no
USE_NATIVE=yes
USE_PETSC=no
USE_PARASAILS=no

ifeq ($(USE_MPI), yes)
  DEFINES += -DUSE_MPI
endif

ifeq ($(USE_NATIVE), yes)
  DEFINES += -DUSE_NATIVE
  SOBJS += COMMON/csr.o COMMON/mat.o COMMON/ssrcsr.o COMMON/mmio.o
  ifeq ($(REAL), double)
    SOBJS += COMMON/ilut.o COMMON/amux.o
  else ifeq ($(REAL), doublecomplex)
    SOBJS += COMMON/zilut.o COMMON/zamux.o
  endif
endif

ifeq ($(USE_PARASAILS), yes)
  ifneq ($(USE_MPI), yes)
    $(error "PARASAILS needs MPI")
  endif
  DEFINES += -DUSE_PARASAILS
  SOBJS += COMMON/parasailsw.o COMMON/csr.o COMMON/ssrcsr.o COMMON/mmio.o
  INCLUDE += -I$(HOME)/local/include/ParaSails
  LIBDIRS += -L$(HOME)/local/lib -lParaSails # Add to current libraries
  LIBS += -lParaSails
endif

ifeq ($(USE_PETSC), yes)
  include ${PETSC_DIR}/lib/petsc/conf/variables
  DEFINES += -DUSE_PETSC
  SOBJS += COMMON/petscw.o COMMON/mmio.o
  INCLUDE += $(PETSC_CCPPFLAGS)
  LIBDIRS += $(PETSC_C_SH_LIB_PATH)
  LIBS += $(PETSC_LIB)
endif


OBJS = $(sort $(SOBJS))

.PHONY: clean veryclean

primme: $(OBJS) $(TOP)/libprimme.a Makefile
	$(CLDR) -o primme $(OBJS) $(LIBDIRS) $(INCLUDE) $(LIBS) $(LDFLAGS) 

.c.o:
	$(CC) $(HIGH_OPT) $(CFLAGS) $(DEFINES) $(INCLUDE) -c $< -o $@

.f.o:
	$(F77) $(HIGH_OPT) $(FFLAGS) -c $< -o $@

clean:  
	rm -f *.o COMMON/*.o $(OBJS)

veryclean:
	@(\
	echo "     "`pwd`;\
	echo "rm -f seq_zprimme *.o core";\
	rm -f seq_dprimme *.o core;\
	)

all: primme
