mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
echo "Run with vector of length 10"
./vector_add 10
echo "Run with vector of length 100"
./vector_add 100
