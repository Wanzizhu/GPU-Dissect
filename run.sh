
export PATH=$PATH:/usr/local/./cuda-12.4/bin/

echo "Compiling $1"
rm -f a.ptx
rm -f a.out
nvcc -arch=sm_80 -ptx  $1 -o a.ptx
nvcc -gencode arch=compute_80,code=sm_80 $1 -o a.out
