import Rotations
import LinearAlgebra
using Test

@testset "Fit ICP" begin


using Posegraph

graph = Graph()

p1 = Float64[1, 0, 0]
p2 = Float64[0, 1, 0]
p3 = Float64[0, 0, 1]


r = rand(Rotations.RotMatrix{3}) # uses Float64 by default
t = Float64[1.1, 2.2, 3.3]
p11 = r * p1 + t
p21 = r * p2 + t
p31 = r * p3 + t

T = Matrix{Float64}(LinearAlgebra.I, 4, 4)

se3_v = VertexSE3("Tdst_src", T)

add_vertex!(graph, se3_v)

pe1 = Point3DistanceError(se3_v, p1, p11)
add_edge!(graph, pe1)

pe2 = Point3DistanceError(se3_v, p2, p21)
add_edge!(graph, pe2)

pe3 = Point3DistanceError(se3_v, p3, p31)
add_edge!(graph, pe3)


solve!(graph)

T = getT(se3_v)


@test T[1:3, 1:3] ≈ r
@test T[1:3, 4] ≈ t

end