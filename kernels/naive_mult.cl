__kernel void multiply(const __global float* A,
                       const __global float* B,
                       __global float* C,
                       const int widthA, 
                       const int widthB) 
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
 
    float result = 0.0;
    for (int i = 0; i < widthA; ++i) 
    {
        result += A[widthA * y + i] * B[widthB * i + x];
    }
 
    C[widthB * y + x] = result;
}

