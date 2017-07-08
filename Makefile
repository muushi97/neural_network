all:
	g++ -std=c++14 -o fuga.exe ./main.cpp ./_activation_function.hpp ./activation_function.hpp ./initializer.hpp ./learning_device.hpp ./perceptron_parameter.hpp ./perceptron.hpp ./ramp_function.hpp ./sigmoid_function.hpp ./softplus_function.hpp -static
	@cp fuga.exe run.exe
	@rm fuga.exe
	@echo finished
