# Motion Capture MTDS Project Files

These are the project files for the Motion Capture (Mocap) section of the
Multi-Task Dynamical System (MTDS) model. This is primarily written in Julia,
but due to the high performance of vanilla GRU/RNN models in PyTorch due to
high performance GPU kernels, many of the final models are written in Python.
These can be found in the fork of Martinez et al. (2017):
[human-motion-prediction-pytorch](https://github.com/ornithos/human-motion-prediction-pytorch).
I'm aware Mike Innes and the developers of Flux.jl are working on this, and
perhaps this could be achieved in Julia in the near future, but for the time
being this is spread across multiple projects -- my apologies.

This project contains a number of utility and processing functions as described
below. In particular, the processing and calculation of model inputs and outputs
in `io.jl` owe quite a lot to [Daniel Holden](http://theorangeduck.com/)'s code,
and perhaps some additions made by [Ian Mason](https://ianxmason.github.io).
In fact, due to the wonders of the [Pycall.jl](https://github.com/JuliaPy/PyCall.jl)
project, some of their original code is called directly, interpreted in its
original python (see `pyfiles` folder).

You'll need to use my utilities package to get all of this to work: [AxUtil.jl](https://github.com/ornithos/AxUtil.jl) (e.g. via `Pkg.add()` with
`PackageSpec(url=...)`) as this project depends on a few functions from there,
although not too many).

Files:

* `io.jl`: in retrospect, a poor choice of name, this refers to the inputs and
outputs of the *model*, not the program. This contains the code to process raw
BVH files (using Dan Holden's `BVH.py`), perform Forward Kinematics and Inverse
Kinematics on trajectories and joints (sometimes), and create the datasets for
model inputs and outputs. The latter (esp. the inputs) have a fairly large
number of options which I played around with to maximise the information given
to the model, but minimise information leakage about style. For the bits I wrote
myself, I hope this is better commented.
  * `input_smooth.jl` contains some additional functionality for `io.jl` for
  smoothing trajectories using cubic B-splines, among other things.
* `mocap_viz.jl`: Visualisation via 3D animation in browser (using
[three.js](https://threejs.org/), via [MeshCat.jl](https://github.com/rdeits/MeshCat.jl)).
Compared to some mocap visualisation (e.g. using Unity or Blender for instance),
this might feel fairly bare bones, but the integration with Julia was too nice
not to play around with.
* `models.jl`: contains the model code for MT-LDS and MT-ORNN architectures. We
were hoping that something quite simple could work well for Mocap data, but in
the end, I found we needed nonlinear models to capture "set pieces" like corners,
and bigger GRUs were helpful at least for smoothness.
* `util.jl` contains a few useful objects: `MyStandardScaler` (in sympathy with
sklearn APIs) for normalising data, and `DataIterator` to iterate through the
training set during training (for instance).

There's a few additional files which are less important to the project such as:
* `torchutil.jl` contains some nice functions to convert the PyTorch models into
native Julia code to avoid running via `PyCall.jl` into PyTorch, but since doing
the latter doesn't have too much overhead, this isn't strictly necessary. Since
the models kept evolving, I don't think this file has kept pace, and should be
considered somewhat experimental.
* `mtdsinf.jl` which contains some functions for performing inference, in
particular the form of population MC (Adaptive Mixture Importance Sampling,
Cappé et al., 2008), but needs `combinedsmcs.jl` which **CURRENTLY IS NOT
AVAILABLE (ping me)**.
* `pretty.jl` is just for making tables in jupyter notebooks and exporting to
latex.





## References

* Martinez et al., "On human motion prediction using recurrent neural networks"
(IEEE CVPR, 2017)
* Tang et al., "Long-Term Human Motion Prediction by Modeling Motion Context and
Enhancing Motion Dynamic" (IJCAI, 2018)
* Cappé et al. "Adaptive importance sampling in general mixture classes."
(Statistics and Computing, 2008)
