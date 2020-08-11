
# OPTS 

CC := gcc
FLAGS := -Wall -Werror -Wextra -pedantic
AR := ar
RM := rm -rf

# SOURCE

CNETDIR := cnet
SOURCES := $(wildcard $(CNETDIR)/*.c)

# BIN PATHS

BDIR := bin
ODIR := $(BDIR)/obj
LDIR := $(BDIR)/lib
XDIR := $(BDIR)/exec
LIBNAME := cnet
LIB := $(LDIR)/lib$(LIBNAME).a

OBJECTS := $(SOURCES:$(CNETDIR)/%.c=$(ODIR)/%.o)

# RANDOM SOURCE

RDNDIR := rdn
RDN_SRC := rdn/main.c
RDN_EXEC := rdn

# LIB COMPILE 

$(ODIR)/%.o: $(CNETDIR)/%.c
	@mkdir -p $(ODIR)
	$(CC) $(FLAGS) -c $? -o $@

$(LIB): $(OBJECTS)
	@mkdir -p $(LDIR)
	$(AR) rcs $@ $^

.PHONY : all
all: $(LIB)

# RDN

$(XDIR)/$(RDN_EXEC): $(LIB) $(RDNDIR)/main.c
	@mkdir -p $(XDIR)
	$(CC) $(FLAGS) -o $@ -l${LIBNAME} -L${LDIR} ${RDN_SRC}

rdn: $(XDIR)/$(RDN_EXEC)

# CLEANS

clean_exec:
	$(RM) $(XDIR)

clean_bin:
	$(RM) $(BDIR)

clean_objects:
	$(RM) $(ODIR)/*.o

clean_lib:
	$(RM) $(LDIR)/*.a

clean: clean_objects clean_lib clean_bin
