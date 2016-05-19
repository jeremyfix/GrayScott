#include <boost/python/extract.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python.hpp>

class GrayScott {

	public:
		GrayScott(std::string param_name, 
			  int N, unsigned int mode, bool measure_fps) {
			
		}

};


BOOST_PYTHON_MODULE(libgrayscott) {

	// tell Boost::Python under what name the array is known
	boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

	// initialize the Numpy C API
	import_array();

	boost::python::class_<WrapperBLC>("blc", boost::python::init<>())
		.def("set_pan", &WrapperBLC::set_pan)
		.def("get_pan", &WrapperBLC::get_pan);

	boost::python::class_<VideoGrabber>("VideoGrabber", boost::python::init<std::stri  ng, unsigned int, unsigned int, PixFormat>())
		.def("grab", &VideoGrabber::grab)
		.def("release", &VideoGrabber::release);

	boost::python::enum_<PixFormat>("PixFormat")
		.value("SGBRG8", SGBRG8)
		.value("SGRBG8", SGRBG8)
		.value("SBGGR8", SBGGR8)
		.value("SRGGB8", SRGGB8);

	//boost::python::def("bayer_to_rgb24", bayer_to_rgb24);
}

