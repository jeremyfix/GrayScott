// g++ -shared -o libgrayscott.so libgrayscott.cc `pkg-config --libs --cflags python` -lboost_python -O3

#include <boost/python/extract.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include <cstdlib>
#include <cstring>

class GrayScott {

    private:

        std::string param_name;
        unsigned int N;

        double k, F;
        double h,Du, Dv, dt, noise;
        double* ut_1, *vt_1, *lut_1, *lvt_1, *ut, *vt;
                      
       void compute_laplacians(void) {
           std::memset(lut_1, 0, N*N*sizeof(double));
           std::memset(lvt_1, 0, N*N*sizeof(double));

       } 

    public:
        GrayScott(std::string param_name, unsigned int N) : param_name(param_name), N(N) {
            if(param_name == std::string("solitons")) {
                k = 0.056;
                F = 0.020;             
            }
            else if(param_name == std::string("worms")) {
                k = 0.0630;
                F = 0.0580;               
            }
            else if(param_name == std::string("spirals")) {
                k = 0.0370;
                F = 0.0060;
            }
            else {
                k = 0.040;
                F = 0.060;
            }

            h = 1e-2;
            Du = 2 * 1e-5 / (h*h);
            Dv = 1e-5 / (h*h);
            dt = 1.0;
            noise = 0.1;
            ut_1 = new double[N * N];
            vt_1 = new double[N * N];
            lut_1 = new double[N * N];
            lvt_1 = new double[N * N];
            ut = new double[N*N];
            vt = new double[N*N];
           
        }

        ~GrayScott(void) {
            delete[] ut_1;
            delete[] vt_1;
            delete[] lut_1;
            delete[] lvt_1;
            delete[] ut;
            delete[] vt;
        }

        void init(void) {
            if(param_name == std::string("spirals")) {
                double* ut_1_ptr = ut_1;
                double* vt_1_ptr = vt_1;
                for(unsigned int i = 0 ; i < N*N; ++i) {
                    *(ut_1_ptr++) = std::rand() / double(RAND_MAX);
                    *(vt_1_ptr++) = std::rand() / double(RAND_MAX);
                }
            }
            else {
                unsigned int dN = N/4;
                std::memset(ut_1, 1, N*N*sizeof(double));
                std::memset(vt_1, 0, N*N*sizeof(double));

                for(int i = -dN ; i <= dN ; ++i)
                    for(int j = -dN ; j <= dN; ++j) {
                        ut_1[(N/2+i)*N + (N/2+j)] = 0.5;
                        vt_1[(N/2+i)*N + (N/2+j)] = 0.25;
                    }               
                
                double* ut_1_ptr = ut_1;
                double* vt_1_ptr = vt_1;
                for(unsigned int i = 0 ; i < N*N; ++i) {
                    *(ut_1_ptr++) += noise * (2.0 * std::rand() / double(RAND_MAX) - 1.);
                    *(vt_1_ptr++) += noise * (2.0 * std::rand() / double(RAND_MAX) - 1.);
                }
 
            }
        }

        PyObject* step(void) {
            // The GrayScott equations read :
            // du/dt = Du Laplacian(u) - u * v^2 + F * (1 - u)
            // dv/dt = Dv Laplacian(v) + u * v^2 - (F + k) * v
            //
            // Discretized in time with Forward Euler :
            // u(t+dt) = u(t) + dt * (Du Laplacian(u(t)) - u(t) * v(t)^2 + F * (1 - u(t))
            // v(t+dt) = v(t) + dt * (Dv Laplacian(v(t)) + u(t) * v(t)^2 - (F + k) * v(t))
            //
            //
            compute_laplacians();

            double* ut_1_ptr = ut_1;
            double* vt_1_ptr = vt_1;
            double* ut_ptr = ut;
            double* vt_ptr = vt;
            double* lut_1_ptr = lut_1;
            double* lvt_1_ptr = lvt_1;
            for(unsigned int i = 0 ; i < N*N ; ++i, ++ut_1_ptr, ++vt_1_ptr, ++ut_ptr, ++vt_ptr, ++lut_1, ++lvt_1) {
                double uvv = (*ut_1_ptr) * (*vt_1_ptr) * (*vt_1_ptr);
                *ut_ptr = *ut_1_ptr + dt * ( Du * (*lut_1_ptr) - uvv + F * (1.0 - (*ut_1_ptr)));
                *vt_ptr = *vt_1_ptr + dt * ( Dv * (*lvt_1_ptr) + uvv - (F + k) * (*vt_1_ptr));
            }
            
            
            

            npy_intp* dims = new npy_intp[2];
            dims[0] = N;
            dims[1] = N;
            PyObject* py_ut = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, ut);
            return py_ut;
        }


};


BOOST_PYTHON_MODULE(libgrayscott) {

    // tell Boost::Python under what name the array is known
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

    // initialize the Numpy C API
    import_array();

    boost::python::class_<GrayScott>("GrayScott", boost::python::init<std::string, unsigned int>())
        .def("init", &GrayScott::init)
        .def("step", &GrayScott::step);
}

