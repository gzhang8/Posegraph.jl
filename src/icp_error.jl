"""
error over T while src and dst pts are not optimized

dst_pt_src_pt::Vector{Float64}: this put two points in one vector
1:3  dst_pt    4:6 src_pt

"""
struct Point3DistanceError <: Edge
    Tdst_src::VertexSE3
    dst_pt_src_pt::Vector{Float64}
    # ceres_cost_func
end

"""
distance error over 3d points for ceres
This error consider se3 T as optimization variable
The two points are considered fixed
This error is designed for ICP
"""
function error_point3_dist_var_se3(user_data, parameters)
    observation3d_dst = user_data[1:3]
    observation3d_src = user_data[4:6]

    q_dst_src = q_a = Rotations.QuatRotation(parameters[1]...) # 4
    trans_dst_src = parameters[2] # 3

    p_src_trans = q_dst_src * observation3d_src
    p_src_trans += trans_dst_src

    residual = observation3d_dst - p_src_trans
    return residual
end

"""
function Point3Error(
    Tdst_src::VertexSE3,
    src_pt::T,
    dst_pt::T
) where T<:Union{Vector{Float64}, Matrix{Float64}}
"""
function Point3DistanceError(
    Tdst_src::VertexSE3{T1},
    src_pt::T2,
    dst_pt::T3
) where {T1, T2<:AbstractArray{<:Real}, T3<:AbstractArray{<:Real}}

    @assert length(src_pt) == 3
    @assert length(dst_pt) == 3
    dst_src_pt::Vector{Float64} = vcat(dst_pt[:], src_pt[:])
    Point3DistanceError(
        Tdst_src,
        dst_src_pt
        # error_point3_dist_var_se3
    )
end

function add_edge!(g::Graph, e::Point3DistanceError)
    Ceres.AddResidualBlock!(
        g.ceres_problem,
        error_point3_dist_var_se3,
        e.dst_pt_src_pt,
        [e.Tdst_src.qwxyz, e.Tdst_src.xyz],
        3)
    @assert haskey(g.vertices, e.Tdst_src.id)

    # set quaternion_parameterization
    Ceres.SetLocalParameterization!(
        g.ceres_problem,
        e.Tdst_src.qwxyz,
        Ceres.quaternion_parameterization)

    push!(g.edges, e)
end
