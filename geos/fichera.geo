// Fichera cube: unit-cube-minus-octant, a classic non-convex, non-smooth
// (single reentrant corner) BEM/FEM benchmark geometry. Built via
// OpenCASCADE boolean subtraction (robust: avoids hand-authored
// point/line/orientation bookkeeping for the 9-face, 21-edge, 14-vertex
// polyhedron that results from removing a corner cube from a cube).
//
// Big cube: [-1,1]^3.  Removed octant: [0,1]x[0,1]x[0,1] (the corner
// nearest (1,1,1)). Resulting solid has 3 full square faces (x=-1,y=-1,
// z=-1), 3 L-shaped hexagonal faces (x=1,y=1,z=1), and 3 new square
// "notch" faces exposed by the cut (x=0,y=0,z=0 restricted to [0,1]^2),
// meeting at a single reentrant vertex at the origin.

SetFactory("OpenCASCADE");

h = 0.25;

Box(1) = {-1, -1, -1, 2, 2, 2};
Box(2) = {0, 0, 0, 1, 1, 1};
BooleanDifference(3) = { Volume{1}; Delete; }{ Volume{2}; Delete; };

Mesh.CharacteristicLengthMax = h;
Mesh.CharacteristicLengthMin = h;

Physical Surface("Gamma") = Boundary{ Volume{3}; };
