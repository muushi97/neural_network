CC = g++
CFLAG = -std=c++14 -static

all:
	$(CC) $(CFLAG) -o fuga.exe ./ramp_function.cpp ./sigmoid_function.cpp ./softplus_function.cpp ./main.cpp ./initializer.cpp ./learning_device.cpp ./perceptron_parameter.cpp ./perceptron.cpp
	@cp fuga.exe run.exe
	@rm fuga.exe
	@echo finished

main.o: main.cpp
	$(CC) $(CFLAGS) -c main.cpp

initializer.o: initializer.cpp initializer.hpp
	$(CC) $(CFLAG) -c initializer.cpp

learning_device.o: learning_device.cpp learning_device.hpp
	$(CC) $(CFLAG) -c learning_device.cpp

perceptron_parameter.o: perceptron_parameter.cpp perceptron_parameter.hpp
	$(CC) $(CFLAG) -c perceptron_parameter.cpp

perceptron.o: perceptron.cpp perceptron.hpp
	$(CC) $(CFLAG) -c perceptron.cpp

ramp_function.o: ramp_function.cpp ramp_function.hpp
	$(CC) $(CFLAG) -c ramp_function.cpp

sigmoid_function.o: sigmoid_function.cpp sigmoid_function.hpp
	$(CC) $(CFLAG) -c sigmoid_function.cpp

softplus_function.o: softplus_function.cpp softplus_function.hpp
	$(CC) $(CFLAG) -c softplus_function.cpp
