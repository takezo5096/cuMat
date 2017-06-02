CC=g++
NVCC=nvcc
CUDA_TOP=/usr/local/cuda
INC=-I$(CUDA_TOP)/include
#LIB=-L$(CUDA_TOP)/lib64 -L./ -lcublas -lcudart -lm


OBJ=softmax_kernel.o mat_log_kernel.o mat_sin_kernel.o mat_cos_kernel.o adam2_kernel.o dropout_kernel.o mat_mul_elementwise_plus_kernel.o mat_sqrt_kernel.o mat_sqrt_d_kernel.o relu_d_kernel.o relu_kernel.o prelu_d_kernel.o prelu_kernel.o sigmoid_d_kernel.o sigmoid_kernel.o tanh_d_kernel.o tanh_kernel.o softmax_cross_entropy_kernel.o mat_sum_kernel.o mat_l2_kernel.o mat_div_kernel.o mat_ones_kernel.o mat_mul_elementwise_kernel.o mat_vec_mul_kernel.o mat_dot_product_kernel.o mat_exp_kernel.o element_wise_clip_kernel.o mat_inverse_kernel.o mat_inverse_d_kernel.o batch_sum_kernel.o vec_to_mat_kernel.o im2col.o pooling.o slice_rows_kernel.o
#OBJ=cuMat.o softmax_kernel.o mat_log_kernel.o mat_sin_kernel.o mat_cos_kernel.o adam2_kernel.o dropout_kernel.o mat_mul_elementwise_plus_kernel.o mat_sqrt_kernel.o mat_sqrt_d_kernel.o relu_d_kernel.o relu_kernel.o prelu_d_kernel.o prelu_kernel.o sigmoid_d_kernel.o sigmoid_kernel.o tanh_d_kernel.o tanh_kernel.o softmax_cross_entropy_kernel.o mat_sum_kernel.o mat_l2_kernel.o mat_div_kernel.o mat_ones_kernel.o mat_mul_elementwise_kernel.o mat_vec_mul_kernel.o mat_dot_product_kernel.o mat_exp_kernel.o element_wise_clip_kernel.o mat_inverse_kernel.o mat_inverse_d_kernel.o batch_sum_kernel.o vec_to_mat_kernel.o im2col.o pooling.o

libcumat.so:$(OBJ)
	gcc -shared -o libcumat.so $(OBJ)
#	gcc -shared -o libcumat.so $(OBJ) $(LIB)

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

mat_sqrt_d_kernel.o: mat_sqrt_d_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c mat_sqrt_d_kernel.cu $(INC)

relu_d_kernel.o: relu_d_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c relu_d_kernel.cu $(INC)

relu_kernel.o: relu_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c relu_kernel.cu $(INC)

prelu_d_kernel.o: prelu_d_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c prelu_d_kernel.cu $(INC)

prelu_kernel.o: prelu_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c prelu_kernel.cu $(INC)

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

mat_vec_mul_kernel.o: mat_vec_mul_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c mat_vec_mul_kernel.cu $(INC)

mat_dot_product_kernel.o: mat_dot_product_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c mat_dot_product_kernel.cu $(INC)

mat_exp_kernel.o: mat_exp_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c mat_exp_kernel.cu $(INC)

element_wise_clip_kernel.o: element_wise_clip_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c element_wise_clip_kernel.cu $(INC)

mat_inverse_kernel.o: mat_inverse_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c mat_inverse_kernel.cu $(INC)

mat_inverse_d_kernel.o: mat_inverse_d_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c mat_inverse_d_kernel.cu $(INC)

vec_to_mat_kernel.o: vec_to_mat_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c vec_to_mat_kernel.cu $(INC)

batch_sum_kernel.o: batch_sum_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c batch_sum_kernel.cu $(INC)

im2col.o: im2col.cu
	$(NVCC) -Xcompiler -fPIC -c im2col.cu $(INC)

pooling.o: pooling.cu
	$(NVCC) -Xcompiler -fPIC -c pooling.cu $(INC)

slice_rows_kernel.o: slice_rows_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c slice_rows_kernel.cu $(INC)

#cuMat.o: cuMat.cpp
#	$(CC) -fPIC -c cuMat.cpp $(INC) -std=c++11


clean:
	rm -f libcumat.so
	rm -f *.o
