This C++ implementation is the reference solution to the shortcut problem, and has been copied, as unmodified as possible, from [here](http://ppc.cs.aalto.fi/ch2/).

## Building

### With CMake
```
cmake -G "Unix Makefiles"
make
```

### (or without CMake)
```
mkdir target
g++ -g -O3 -march=native -std=c++17 main.cpp -o target/shortcut
```

### Run

```
./target/shortcut
```
