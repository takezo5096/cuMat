CC=g++
NVCC=nvcc
CUDA_TOP=/usr/local/cuda
INC=-I$(CUDA_TOP)/include
#LIB=-L$(CUDA_TOP)/lib64 -L./ -lcublas -lcudart -lm


#OBJ= adam2_kernel.o adam_kernel.o dropout_kernel.o matcos_kernel.o matdiv_kernel.o matl2_kernel.o matlog_kernel.o matmod_kernel.o matmul2_kernel.o matmul2_plus_kernel.o matones_kernel.o matsin_kernel.o matsqrt_kernel.o matsum_kernel.o relu_d_kernel.o relu_kernel.o sigmoid_d_kernel.o sigmoid_kernel.o softmax_cross_entropy_kernel.o softmax_kernel.o tanh_d_kernel.o tanh_kernel.o
OBJ=softmax_kernel.o mat_log_kernel.o mat_sin_kernel.o mat_cos_kernel.o adam2_kernel.o dropout_kernel.o mat_mul_elementwise_plus_kernel.o mat_sqrt_kernel.o relu_d_kernel.o relu_kernel.o sigmoid_d_kernel.o sigmoid_kernel.o tanh_d_kernel.o tanh_kernel.o softmax_cross_entropy_kernel.o mat_sum_kernel.o mat_l2_kernel.o mat_div_kernel.o mat_ones_kernel.o mat_mul_elementwise_kernel.o

libcumat.so:$(OBJ)
	gcc -shared -o libcumat.so $(OBJ)

softmax_kernel.o: softmax_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c softmax_kernel.cu $(INC)

mat_log_kernel.o: mat_log_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c mat_log_kernel.cu $(INC)

mat_sin_kernel.o: mat_sin_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c mat_sin_kernel.cu $(INC)

mat_cos_kernel.o: mat_cos_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c mat_cos_kernel.cu $(INC)

adam2_kernel.o: adam2_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c adam2_kernel.cu $(INC)

dropout_kernel.o: dropout_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c dropout_kernel.cu $(INC)

mat_mul_elementwise_plus_kernel.o: mat_mul_elementwise_plus_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c mat_mul_elementwise_plus_kernel.cu $(INC)

mat_sqrt_kernel.o: mat_sqrt_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c mat_sqrt_kernel.cu $(INC)

relu_d_kernel.o: relu_d_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c relu_d_kernel.cu $(INC)

relu_kernel.o: relu_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c relu_kernel.cu $(INC)

sigmoid_d_kernel.o: sigmoid_d_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c sigmoid_d_kernel.cu $(INC)

sigmoid_kernel.o: sigmoid_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c sigmoid_kernel.cu $(INC)

tanh_d_kernel.o: tanh_d_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c tanh_d_kernel.cu $(INC)

tanh_kernel.o: tanh_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c tanh_kernel.cu $(INC)

softmax_cross_entropy_kernel.o: softmax_cross_entropy_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c softmax_cross_entropy_kernel.cu $(INC)

mat_sum_kernel.o: mat_sum_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c mat_sum_kernel.cu $(INC)

mat_l2_kernel.o: mat_l2_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c mat_l2_kernel.cu $(INC)

mat_div_kernel.o: mat_div_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c mat_div_kernel.cu $(INC)

mat_ones_kernel.o: mat_ones_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c mat_ones_kernel.cu $(INC)

mat_mul_elementwise_kernel.o: mat_mul_elementwise_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c mat_mul_elementwise_kernel.cu $(INC)


clean:
	rm -f libcumat.so
	rm -f mat_*.o
