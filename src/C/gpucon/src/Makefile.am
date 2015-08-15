
EXT = cu

OBJS = gpu_div_by_sd.o gpu_matmul.o gpu_matmul_div.o 

INCLUDE = gpu_div_by_sd.h gpu_matmul.h gpu_matmul_div.h 

TARGETS = libgpucon.so

all: $(TARGETS)

$(TARGETS): $(OBJS)
	$(NVCC) $(CUDA_LDFLAGS) $(CUDA_LIBS) $(OBJS) -o $@

$(OBJS): %.o: %.$(EXT)
	$(NVCC) $(CUDA_CFLAGS) -c $^ -o $@

clean:
	rm -rf *o

install: all
	install -d $(DESTDIR)/lib
	install $(TARGETS) $(DESTDIR)/lib
	install -d $(DESTDIR)/include
	cp $(INCLUDE) $(DESTDIR)/include

