CXX = g++
CFLAGS = -std=c++11 -O3
INCDIR = -I/usr/include \
         -I/usr/local/cuda/include
LDFLAGS = 
OBJS = main.o
NVOBJS = /usr/lib/x86_64-linux-gnu/libcudnn.so \
         /usr/local/cuda/lib64/libcudart.so
NVCC = nvcc
NAME = main.out
CUDA_MEMCHECL = cuda-memcheck

run: $(NAME)
	$(CUDA_MEMCHECL) ./$(NAME)
$(NAME): $(OBJS)
	$(NVCC) -o $(NAME) $(OBJS) $(NVOBJS)

main.o: main.cu
	$(NVCC) $(INCDIR) $(CFLAGS) -c $<

clean:
	$(RM) $(NAME) $(OBJS)
