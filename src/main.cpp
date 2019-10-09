#include <bm3d.h>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;

py::array_t<float>
bm3d(py::array_t<float, py::array::c_style | py::array::forcecast> img,
     float sigma) {

  auto img_buf = img.request();

  unsigned int channels;
  if (img_buf.ndim == 2) {
    channels = 1;
  } else if (img_buf.ndim == 3) {
    channels = img_buf.shape[2];
  } else {
    throw std::invalid_argument("img dimensions must be 2 or 3.");
  }

  std::cout << "img_buf shape: " << img_buf.shape[0] << ", " << img_buf.shape[1]
            << std::endl;
  std::cout << "img_buf strides: " << img_buf.strides[0] << ", "
            << img_buf.strides[1] << std::endl;

  unsigned int height = img_buf.shape[0];
  unsigned int width = img_buf.shape[1];

  std::vector<float> img_noisy((float *)img_buf.ptr,
                               (float *)img_buf.ptr + img_buf.size);
  std::cout << "img_noisey size: " << img_noisy.size() << std::endl;

  std::vector<float> img_basic(img_buf.size);
  std::vector<float> img_denoised(img_buf.size);

  run_bm3d(sigma,        // sigma
           img_noisy,    // img_noisy
           img_basic,    // img_basic
           img_denoised, // img_denoised
           width,        // width
           height,       // height
           channels,     // chnls
           false,        // useSD_h
           false,        // useSD_w
           5,            // tau_2D_hard
           4,            // tau_2D_wien
           2,            // color_space
           0,            // patch_size
           0,            // nb_threads
           true          // verbose
  );

  auto result =
      py::array_t<float>(img_buf.shape, img_buf.strides, img_denoised.data());

  std::cout << "result size: " << result.size() << std::endl;
  return result;
}

py::array_t<double>
f(py::array_t<double, py::array::c_style | py::array::forcecast> img) {
  auto img_buf = img.request();
  std::cout << "img_buf shape: " << img_buf.shape[0] << ", " << img_buf.shape[1]
            << std::endl;
  std::cout << "img_buf shape: " << img_buf.strides[0] << ", "
            << img_buf.strides[1] << std::endl;

  std::vector<double> foo_vec((double *)img_buf.ptr,
                              (double *)img_buf.ptr + img_buf.size);

  std::cout << "foo_vec size: " << foo_vec.size() << std::endl;

  foo_vec[0] = 2020.0;

  auto array =
      py::array_t<double>(img_buf.shape, img_buf.strides, foo_vec.data());
  foo_vec[0] = 1010.0;
  return array;
}

PYBIND11_MODULE(pybind_bm3d, m) {
  m.doc() = R"pbdoc(
        BM3D binding using pybind11
        ---------------------------

        .. currentmodule:: pybind_bm3d

        .. autosummary::
           :toctree: _generate

           bm3d
    )pbdoc";

  m.def("bm3d", &bm3d, R"pbdoc(
        Apply the BM3D image to a noisy input image

        Parameters
        ----------
        img : np.array of type float
            The noisy input image. Will be coerced to be C_ALIGNED.
            May have dimensions HxWxC or HxW (Height, Width, Channels).
        sigma : float
            An estimate of the standard deviation of the noise.

        Returns
        -------
        np.array of type float
            Denoised image with the same shape as the input `img`.

        .. _More information:
            Marc Lebrun,
	    An Analysis and Implementation of the BM3D Image Denoising Method,
	    Image Processing On Line, 2 (2012), pp. 175–213.
	    https://doi.org/10.5201/ipol.2012.l-bm3d
            http://www.ipol.im/pub/art/2012/l-bm3d/

    )pbdoc");

  m.def("f", &f);

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
