# Phony target to avoid conflicts with files named "clean"
.PHONY: clean

# Makefile for your C++ project

# Compiler and compiler flags
CXX = g++
CXXFLAGS = -std=c++11 -I/usr/local/include/eigen3 -I/usr/local -Ilib/eigen-3.4.0 -I/usr/local/include -Ilib/

LDFLAGS = -L/usr/local/lib -lspdlog

# Source files and target executable
SRCS = cxx/geometry_functions.cxx main.cxx
TARGET = main

# Build rule
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

# Clean rule
clean:
	rm -f $(TARGET)


# python parts


test:
	pytest

cpython:
	g++ -o sgd_wrapper sgd_wrapper.cpp -I$(python3 -m pybind11 --includes) $(python3-config --ldflags)
