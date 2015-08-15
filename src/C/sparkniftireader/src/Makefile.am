
EXT = c

OBJS = sparkniftireader.o

TARGETS = libsparkniftireader.so

all: $(TARGETS)

# $(TARGETS): $(OBJS)
# 	$(CC) $(NIFTI_LDFLAGS) $(NIFTI_LIBS) $(OBJS) -o $@
$(TARGETS):
	$(CC) $(NIFTI_CFLAGS) $(NIFTI_LDFLAGS) $(NIFTI_INCLUDE) -o $@ sparkniftireader.c $(NIFTI_LIBS)

# $(OBJS): %.o: %.$(EXT)
# 	$(CC) $(NIFTI_CFLAGS) $(NIFTI_INCLUDE) -c $^ -o $@

clean:
	rm -rf *o

install: all
	install -d $(DESTDIR)/lib
	install $(TARGETS) $(DESTDIR)/lib
