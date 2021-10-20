module Posegraph


import LinearAlgebra
import Ceres
using Ceres: LossFunction, EmptyLoss, HuberLoss
import Rotations


export add_vertex!, add_edge!,

       # major part
       Graph, Edge, Vertex, VertexSE3, Point3DistanceError,
       SE3Edge,
       getT, setT!,
       #add_edge!, add_vertex!,
       solve!,

       # SBA
       VertexPoint3d, EdgeSE3ProjectXYZ, EdgeStereoSE3ProjectXYZ



include("base.jl")
include("converter.jl")
include("se3.jl")
include("icp_error.jl")

include("VertexPoint3d.jl")
include("MonoProjectXYZ.jl")
include("StereoProjectXYZ.jl")

include("switchable.jl")

include("gps3d.jl")





end # module
