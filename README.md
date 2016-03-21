# Eigenvalues

PyQt GUI for Laplace eigenvalue/eigenfunction computations using [FEniCS](http://fenicsproject.org) as finite element backend.

Two and three dimensional domains can be generated, visualized, opened in new window and interactively rotated.

<img source="doc/eig1.png" width="400"/>

Solutions can also be visualized, opened in new windows, and interactively rotated. Level sets can be shown in 2D and 3D, in addition to boundary values in 3D and warp function plots for 2D.

<img source="doc/eig2.png" width="400"/>

### Requirements

- Working [FEniCS](http://fenicsproject.org) installation is absolutely necessary.
- On a Mac one can simply download `Eigenvalues.app` from `osx/` subfolder and put it in `/Applications`. No need to put FEniCS in the `.profile`, as mentioned on [FEniCS installation page](http://fenicsproject.org/download/osx_details.html).
- Otherwise Python packages `billiard`, and `PyQt4` (or 5) must be installed.

### Features
- GUI is independent from solving engine, so it is be possible to run computations without `PyQt`.
- Domain generation supports semi rigid parameters parsing. Most input fileds support valid `python` code, as well as lists of numbers separated by spaces.
- Domains can be transformed using a large class of mesh transformations, including complex, conformal, and starlike maps.
- Dirichlet, Neumann, Robin and Steklov boundary conditions are supported. As well as arbitrary mixtures of those, and interior Dirichlet conditions.
- FEM engine supports conforming (orders 1-3) and nonconforming (order 1) elements. Targeting eigenvalues around a specified number is also possible.
- Eigenvalues can be rescaled using the area, perimeter, and moment of inertia. Sums and other spectral functionals can also be computed.
- Eigenfunctions can symmetrized with respect to any isometry of a square/cube, assuming a domain is appropriately symmetric.
