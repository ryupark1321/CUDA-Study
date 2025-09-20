rm -rf build
rm demo*
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
mv demo ../
mv demo_safe ../
echo "Running demo"
cd ..
./demo > demo.txt
echo "Running demo_safe"
./demo_safe > demo_safe.txt
