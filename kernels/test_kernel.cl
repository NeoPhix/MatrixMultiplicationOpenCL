__kernel void matrix_sum(__global const double *A, __global const double *B, __global double *C) {
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);
 
    // Do the operation
    C[i] = A[i] + B[i];
}