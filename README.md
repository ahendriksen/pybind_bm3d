# pybind_bm3d

This module implements a python binding for the bm3d source code by Marc Lebrun.

Marc Lebrun,
An Analysis and Implementation of the BM3D Image Denoising Method,
Image Processing On Line, 2 (2012), pp. 175–213.
[](https://doi.org/10.5201/ipol.2012.l-bm3d)
[](http://www.ipol.im/pub/art/2012/l-bm3d/)



## Installation


```bash
conda install -c aahendriksen pybind_bm3d
```

## Building the documentation

Documentation for the example project is generated using Sphinx. Sphinx has the
ability to automatically inspect the signatures and documentation strings in
the extension module to generate beautiful documentation in a variety formats.
The following command generates HTML-based reference documentation; for other
formats please refer to the Sphinx manual:

 - `pip install recommonmark sphinx_rtd_theme sphinx`
 - `cd docs`
 - `make html`


## License

Pybind11 is provided under a BSD-style license that can be found in
the LICENSE file. By using, distributing, or contributing to this
project, you agree to the terms and conditions of this license.

## Test call

```python
import pybind_bm3d as m
import numpy as np

sigma = 1.0
noisy_img = np.random.normal(0, sigma, size=(100, 100)).astype(np.float32)

denoised_img = m.bm3d(noisy_img, sigma)
```
