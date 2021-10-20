using Posegraph
# import Makie
# import SLAMData
using LinearAlgebra
using Test

@testset "Fit GPS 3D" begin

g = Graph()

T = Matrix{Float64}(I, 4, 4)
# T[1:3, 4] = [1.0, 2.1, 3.3]
se3_v = VertexSE3("1", T, fixed=false)
add_vertex!(g, se3_v)

gps_reading = [1.0, 2.1, 3.3]

info = Matrix{Float64}(I, 3, 3)
gps_e1 = Posegraph.GPS3DError(se3_v, gps_reading, info, info_is_sqrt=false, is_Twc=false)

add_edge!(g, gps_e1)

solve!(g)

T = getT(se3_v)

# display(T)

@test T[1:3, 1:3] ≈ Matrix{Float64}(I, 3, 3)
@test T[1:3, 4] ≈ gps_reading

end