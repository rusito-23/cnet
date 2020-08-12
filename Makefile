
# ----------------------- #
#          OPTS	 
# ----------------------- #


CC := gcc
FLAGS := -Wall -Werror -Wextra -pedantic
AR := ar
RM := rm -rf


# ----------------------- #
# 	BIN PATHS
# ----------------------- #


BDIR := bin
ODIR := $(BDIR)/obj
LDIR := $(BDIR)/lib
XDIR := $(BDIR)/exec


# ----------------------- #
# 	CNET LIB
# ----------------------- #


CNET := cnet
CNET_SDIR := $(CNET)/src
CNET_IDIR := $(CNET)/include
CNET_LIB := $(LDIR)/lib$(CNET).a
CNET_SRC := $(wildcard $(CNET_SDIR)/*.c)
CNET_OBJ := $(CNET_SRC:$(CNET_SDIR)/%.c=$(ODIR)/%.o)

$(ODIR)/%.o: $(CNET_SDIR)/%.c
	@mkdir -p $(ODIR)
	$(CC) $(FLAGS) -I$(CNET_IDIR) -c $? -o $@

$(CNET_LIB): $(CNET_OBJ)
	@mkdir -p $(LDIR)
	$(AR) rcs $@ $^

$(CNET): $(CNET_LIB) 


# ----------------------- #
# 	  TESTS
# ----------------------- #


TEST := test
TEST_SRC := $(TEST)/*.c

$(XDIR)/$(TEST): $(CNET_LIB) $(TEST_SRC)
	@mkdir -p $(XDIR)
	$(CC) $(FLAGS) -o $@ -I${CNET_IDIR} -l${CNET} -L${LDIR} ${TEST_SRC}

test: $(XDIR)/$(TEST)


# ----------------------- #
#	  CLEANS
# ----------------------- #

clean_obj:
	$(RM) $(ODIR)

clean_lib:
	$(RM) $(LDIR)

clean_exec:
	$(RM) $(XDIR)

clean: clean_obj clean_lib clean_exec
	$(RM) $(BDIR)
