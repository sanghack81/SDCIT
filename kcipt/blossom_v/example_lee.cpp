#define BLOSSOM_V_MODULE

#include <Python.h>
#include <stdio.h>
#include "PerfectMatching.h"
#include "GEOM/GeomPerfectMatching.h"
#include <iostream>


static PyObject *blossom_v(PyObject *self, PyObject *args)
{
	struct PerfectMatching::Options options;
	PyObject *edges_py;
	PyObject *weights_py;
	PyObject *matching;
    PerfectMatching *pm;

	int i, e, len, node_num, edge_num;
	int *edges;
	int *weights;
	int *match;

    options.verbose = false;

	if (!PyArg_ParseTuple(args, "iiOO", &node_num, &edge_num, &edges_py, &weights_py)) {
        return NULL;
    }

    len = PyTuple_Size(edges_py);
    edges= new int[len];
    while (len--) {
        edges[len] = (int) PyLong_AsLong(PyTuple_GetItem(edges_py, len));
    }

    len = PyTuple_Size(weights_py);
    weights= new int[len];
    while (len--) {
        weights[len] = (int) PyLong_AsLong(PyTuple_GetItem(weights_py, len));
    }


    pm = new PerfectMatching(node_num, edge_num);
    for (e=0; e<edge_num; e++){
        pm->AddEdge(edges[2*e], edges[2*e+1], weights[e]);
    }
    pm->options = options;
    pm->Solve();

    matching = PyList_New(0);
    for (i=0; i<node_num; i++){
		PyList_Append(matching, Py_BuildValue("i",pm->GetMatch(i)));
	}

    delete pm;
    delete [] edges;
    delete [] weights;

    return matching;
}


static PyMethodDef module_functions[] = {
		{"blossom_v", blossom_v, METH_VARARGS, "Execute a shell command."},
		{NULL, NULL, 0, NULL}
	};


PyMODINIT_FUNC
PyInit_py_blossom_v(void)
{
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "py_blossom_v",     /* m_name */
        "This is a module",  /* m_doc */
        -1,                  /* m_size */
        module_functions,    /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };
	return PyModule_Create(&moduledef);
}

int
main(int argc, char *argv[])
{
    wchar_t progname[FILENAME_MAX + 1];
    mbstowcs(progname, argv[0], strlen(argv[0]) + 1);

	Py_SetProgramName(progname);
	Py_Initialize();
	PyInit_py_blossom_v();
}