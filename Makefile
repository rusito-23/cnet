
# ----------------------- #
#          OPTS	 
# ----------------------- #


CC := gcc
CFLAGS := -Wall -Werror -Wextra -pedantic
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
	$(CC) $(CFLAGS) -I$(CNET_IDIR) -c $? -o $@

$(CNET_LIB): $(CNET_OBJ)
	@mkdir -p $(LDIR)
	$(AR) rcs $@ $^

$(CNET): $(CNET_LIB) 


# ----------------------- #
# 	  TESTS
# ----------------------- #


TEST := test
TEST_SDIR := $(TEST)

$(XDIR)/%.tests: $(TEST_SDIR)/%.c $(CNET_LIB)
	@mkdir -p $(XDIR)
	$(CC) $(CFLAGS) -o $@ -I$(CNET_IDIR) -l$(CNET) -L$(LDIR) $<

integration-tests: $(XDIR)/integration.tests


# ----------------------- #
#	  MNIST
# ----------------------- #

MNIST := mnist
MNIST_SDIR := $(MNIST)
MNIST_SRC := $(wildcard $(MNIST)/*.c)
MNIST_IN := $(wildcard $(MNIST)/*.h)

$(XDIR)/mnist.%: $(MNIST_SDIR)/%.c $(MNIST_IN) $(CNET_LIB)
	@mkdir -p $(XDIR)
	$(CC) $(CFLAGS) -o $@ -I$(CNET_IDIR) -l$(CNET) -L$(LDIR) $<

mnist-train: $(XDIR)/mnist.train
mnist-predict: $(XDIR)/mnist.predict


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
