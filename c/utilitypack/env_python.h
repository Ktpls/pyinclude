#pragma once
#ifdef _DEBUG
#define DEBUG_CANCELED
#undef _DEBUG
#endif
#include <Python.h>
#ifdef DEBUG_CANCELED
#define _DEBUG
#undef DEBUG_CANCELED
#endif

class PythonEnvAutoInitializer {
public:
	PythonEnvAutoInitializer() {
		Py_Initialize();
	}
	~PythonEnvAutoInitializer() {
		Py_Finalize();
	}
} _pythonEnvAutoInitializer = PythonEnvAutoInitializer();