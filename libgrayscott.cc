// g++ -shared -o libgrayscott.so libgrayscott.cc `pkg-config --libs --cflags python` -lboost_python -O3 -fPIC

#include <boost/python/extract.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include <cstdlib>
#include <cstring>
#include <iostream>

class GrayScott {

private:

  std::string param_name;
  unsigned int width, height;

  double d, dt;
  double k, F;
  double h,Du, Dv, noise;
  double* ut_1, *vt_1, *lut_1, *lvt_1, *ut, *vt;

  void compute_laplacian(double* values, double* laplacian) {

    // For efficiently computing the laplacian we will move 5 pointers on values
    // pointing : on the pixel, on its north, on its east, on its west, on its south
    double* v_ptr, *vN_ptr, *vE_ptr, *vS_ptr, *vW_ptr;
    double* l_ptr;

    // We handle the corners
    // #oooooo#
    // oooooooo
    // oooooooo
    // oooooooo
    // #oooooo#
    laplacian[0] = -(4 * values[0]) + (values[(height-1)*width] + values[1] + values[width] + values[(width-1)]);// top left
    laplacian[width-1] = -(4 * values[width-1]) + (values[(height-1)*width + (width-1)] + values[0] + values[width + (width-1)] + values[width-2]); // top right
    laplacian[(height-1)*width] = -(4 * values[(height-1)*width]) + (values[(height-2)*width] + values[(height-1)*width + 1] + values[0] + values[(height-1)*width + (width-1)]); // bottom left
    laplacian[(height-1)*width + (width-1)] = -(4 * values[(height-1)*width + (width-1)]) + (values[(height-2)*width + (width-1)]+values[(height-1)*width] + values[width-1] + values[(height-1)*width + (width-2)]); // bottom right

    // We handle the borders
    // o########o
    // oooooooooo
    // oooooooooo
    // oooooooooo
    // oooooooooo
    v_ptr = values + 1;
    vN_ptr = values + (height-1)*width + 1;
    vE_ptr = values + 2;
    vS_ptr = values + width + 1;
    vW_ptr = values;
    l_ptr = laplacian + 1;
    for(unsigned int i = 1 ; i < width-1 ; ++i, ++v_ptr, ++vN_ptr, ++vE_ptr, ++vS_ptr, ++vW_ptr, ++l_ptr) 
      *l_ptr = (*vN_ptr + *vE_ptr + *vS_ptr + *vW_ptr) - ((*v_ptr) * 4);

    // oooooooooo
    // oooooooooo
    // oooooooooo
    // oooooooooo
    // o########o
    v_ptr = values + (height-1)*width + 1;
    vN_ptr = values + (height-2)*width + 1;
    vE_ptr = values + (height-1)*width + 2;
    vS_ptr = values + 1 ;
    vW_ptr = values + (height-1)*width;
    l_ptr = laplacian + (height-1)*width + 1;
    for(unsigned int i = 1 ; i < width-1 ; ++i, ++v_ptr, ++vN_ptr, ++vE_ptr, ++vS_ptr, ++vW_ptr, ++l_ptr) 
      *l_ptr = (*vN_ptr + *vE_ptr + *vS_ptr + *vW_ptr) - ((*v_ptr) * 4);

    // oooooooooo
    // #ooooooooo
    // #ooooooooo
    // #ooooooooo
    // #ooooooooo
    // oooooooooo
    v_ptr = values + width;
    vN_ptr = values ;
    vE_ptr = values + width + 1;
    vS_ptr = values + 2*width ;
    vW_ptr = values + width + (width-1);
    l_ptr = laplacian + width;
    for(unsigned int i = 1 ; i < height-1 ; ++i, v_ptr+=width, vN_ptr+=width, vE_ptr+=width, vS_ptr+=width, vW_ptr+=width, l_ptr+=width) 
      *l_ptr = (*vN_ptr + *vE_ptr + *vS_ptr + *vW_ptr) - ((*v_ptr) * 4);

 
    // oooooooooo
    // ooooooooo#
    // ooooooooo#
    // ooooooooo#
    // ooooooooo#
    // oooooooooo
    // 
    v_ptr = values + width + (width-1);
    vN_ptr = values + (width-1);
    vE_ptr = values + width;
    vS_ptr = values + 2*width + (width-1) ;
    vW_ptr = values + width + (width-2);
    l_ptr = laplacian + width + (width-1);
    for(unsigned int i = 1 ; i < height-1 ; ++i, v_ptr+=width, vN_ptr+=width, vE_ptr+=width, vS_ptr+=width, vW_ptr+=width, l_ptr+=width) 
      *l_ptr = (*vN_ptr + *vE_ptr + *vS_ptr + *vW_ptr) - ((*v_ptr) * 4);

    // We handle the region inside the array exlucding a border of size 1,
    // i.e. the pixels # below
    // oooooooooo
    // o########o
    // o########o
    // o########o
    // oooooooooo
    v_ptr = values + (1*width + 1);
    vN_ptr = values + 1;
    vE_ptr = values + (1*width + 2);
    vS_ptr = values + (2*width + 1);
    vW_ptr = values + (1*width + 0);

    l_ptr = laplacian + (1*width + 1);
    for(unsigned int i = 1; i < height-1; ++i) {
      for(unsigned int j = 1 ; j < width-1 ; ++j, ++vN_ptr, ++vE_ptr, ++vS_ptr, ++vW_ptr, ++v_ptr, ++l_ptr) 
	*l_ptr = (*vN_ptr + *vE_ptr + *vS_ptr + *vW_ptr) - ((*v_ptr) * 4);

      // For switching to the next line we must move the pointers forward by 2 pixels
      v_ptr += 2;
      l_ptr += 2;
      vN_ptr += 2;
      vE_ptr += 2;
      vS_ptr += 2;
      vW_ptr += 2;
    }
  } 

public:
  GrayScott(std::string param_name, unsigned int width, unsigned int height, double d, double dt) : param_name(param_name), width(width), height(height), dt(dt) {
    if(param_name == std::string("solitons")) {
      k = 0.056;
      F = 0.020;             
    }
    else if(param_name == std::string("worms")) {
      k = 0.0630;
      F = 0.0580;               
    }
    else if(param_name == std::string("spirals")) {
      k = 0.050;
      F = 0.018;
    }
    else if(param_name == std::string("uskate")) {
      k = 0.06093;
      F = 0.0620;
    }
    else {
      k = 0.040;
      F = 0.060;
    }
    h = d / (double)width;
    Du = 2.0 * 1e-5 / (h*h);
    Dv = 1.0 * 1e-5 / (h*h);
    noise = 0.2;

    ut_1 = new double[width*height];
    std::fill(ut_1, ut_1 + width*height, 0.0);

    vt_1 = new double[width*height];
    std::fill(vt_1, vt_1 + width*height, 0.0);

    lut_1 = new double[width*height];
    std::fill(lut_1, lut_1 + width*height, 0.0);

    lvt_1 = new double[width*height];
    std::fill(lvt_1, lvt_1 + width*height, 0.0);

    ut = new double[width*height];
    std::fill(ut, ut + width*height, 0.0);

    vt = new double[width*height];
    std::fill(vt, vt + width*height, 0.0);
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
    double *ut_1_ptr, *vt_1_ptr;

    std::fill(ut_1, ut_1 + width*height, 1.0);
    std::fill(vt_1, vt_1 + width*height, 0.0);

    int dN = std::min(width, height)/4;

    for(int i = -dN/2 ; i <= dN/2 ; ++i) {
      for(int j = -dN/2 ; j <= dN/2; ++j) {
	ut_1[(height/2+i)*width + (width/2+j)] = 0.5;
	vt_1[(height/2+i)*width + (width/2+j)] = 0.25;
      }               
    }

    ut_1_ptr = ut_1;
    vt_1_ptr = vt_1;
    for(unsigned int i = 0 ; i < width*height; ++i, ++ut_1_ptr, ++vt_1_ptr) {
      *ut_1_ptr += noise * (2.0 * (std::rand() / double(RAND_MAX)) - 1.);
      if(*ut_1_ptr <= 0.)
	*ut_1_ptr = 0.;
      *vt_1_ptr += noise * (2.0 * (std::rand() / double(RAND_MAX)) - 1.);
      if(*vt_1_ptr <= 0.)
	*vt_1_ptr = 0.;
    }

  }

  void step(void) {
    // The GrayScott equations read :
    // du/dt = Du Laplacian(u) - u * v^2 + F * (1 - u)
    // dv/dt = Dv Laplacian(v) + u * v^2 - (F + k) * v
    //
    // Discretized in time with Forward Euler :
    // u(t+dt) = u(t) + dt * (Du Laplacian(u(t)) - u(t) * v(t)^2 + F * (1 - u(t))
    // v(t+dt) = v(t) + dt * (Dv Laplacian(v(t)) + u(t) * v(t)^2 - (F + k) * v(t))
    //
    //
    //
            
    compute_laplacian(ut_1, lut_1);
    compute_laplacian(vt_1, lvt_1);
            
    double* ut_1_ptr = ut_1;
    double* vt_1_ptr = vt_1;
    double* ut_ptr = ut;
    double* vt_ptr = vt;
    double* lut_1_ptr = lut_1;
    double* lvt_1_ptr = lvt_1;

    //std::cout << dt << "," << Du << "," << Dv << "," << std::endl;
    for(unsigned int i = 0 ; i < width*height ; ++i, ++ut_1_ptr, ++vt_1_ptr, ++ut_ptr, ++vt_ptr, ++lut_1_ptr, ++lvt_1_ptr) {
      double uvv = (*ut_1_ptr) * (*vt_1_ptr) * (*vt_1_ptr);
      *ut_ptr = *ut_1_ptr + dt * ( Du * (*lut_1_ptr) - uvv + F * (1.0 - (*ut_1_ptr)));
      *vt_ptr = *vt_1_ptr + dt * ( Dv * (*lvt_1_ptr) + uvv - (F + k) * (*vt_1_ptr));
    }

    double* tmp;
    tmp = ut;
    ut = ut_1;
    ut_1 = tmp;

    tmp = vt;
    vt = vt_1;
    vt_1 = tmp;
  }

  PyObject* get_ut(void) {
    npy_intp dims[2];
    dims[0] = height;
    dims[1] = width;
    return PyArray_SimpleNewFromData( 2, dims, NPY_DOUBLE, ut );
  }


};


BOOST_PYTHON_MODULE(libgrayscott) {

  // tell Boost::Python under what name the array is known
  boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

  // initialize the Numpy C API
  import_array();

  boost::python::class_<GrayScott>("GrayScott", boost::python::init<std::string, unsigned int, unsigned int, double, double>())
    .def("init", &GrayScott::init)
    .def("step", &GrayScott::step)
    .def("get_ut", &GrayScott::get_ut);
}

