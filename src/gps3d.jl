"""
error over T for a GPS measurement

gps3d::Vector{Float64}: gps reading, len=12(3+9): x, y, z, then col major info
mat

"""
struct GPS3DError <: Edge
    v_wc::VertexSE3
    gps3d::Vector{Float64}
    is_Twc::Bool
    # ceres_cost_func
end

"""
distance error over GPS measurements for ceres

This error consider se3 Twc as optimization variable

user_data is GPS measurements
"""
function error_gps_3d(user_data, parameters)
    # TODO, disable after debug because there is a check before
    @assert length(user_data) == 3+9
    gps_observation_3d = user_data[1:3]
    information_sqrt = reshape(user_data[4:12], (3, 3))

    # q_dst_src = q_a = Rotations.QuatRotation(parameters[1]...) # 4
    trans_Twc = parameters[1] # 3

    residuals = gps_observation_3d - trans_Twc
    residuals = information_sqrt * residuals
    return residuals
end



"""
distance error over GPS measurements for ceres

This error consider se3 Tcw as optimization variable

user_data is GPS measurements
"""
function error_gps_3d_Tcw(user_data, parameters)
    # TODO, disable after debug because there is a check before
    @assert length(user_data) == 3+9
    gps_observation_3d = user_data[1:3]
    information_sqrt = reshape(user_data[4:12], (3, 3))

    q_Tcw = Rotations.QuatRotation(parameters[1]...) # 4
    trans_Tcw = parameters[2] # 3

    trans_Twc = - inv(q_Tcw) * trans_Tcw

    residuals = gps_observation_3d - trans_Twc
    residuals = information_sqrt * residuals
    return residuals
end


"""
function GPS3DError(
    v_wc::VertexSE3,
    gps3d::T1,
) where T1<:AbstractArray{<:Real}
"""
function GPS3DError(
    v_wc::VertexSE3,
    gps3d::T1,
    information::T2;
    info_is_sqrt::Bool=false,
    is_Twc::Bool=true
) where {T1<:AbstractArray{<:Real}, T2<:AbstractMatrix{<:Real}}

    @assert length(gps3d) == 3
    @assert size(information) == (3,3)

    if !info_is_sqrt
        # C = LinearAlgebra.cholesky(information)
        # information = Matrix(C.L)
        information = sqrt(information)
    end

    # gps3d_f64::Vector{Float64} = if eltype(T1) == Float64
    #     gps3d
    # else
    #     convert(Vector{Float64}, gps3d)
    # end

    gps3d_f64::Vector{Float64} = vcat(gps3d, information[:])

    GPS3DError(v_wc, gps3d_f64, is_Twc)
end

function add_edge!(g::Graph, e::GPS3DError)
    if e.is_Twc
        Ceres.AddResidualBlock!(
            g.ceres_problem,
            error_gps_3d,
            e.gps3d,
            [e.v_wc.xyz],
            3
        )
    else
        Ceres.AddResidualBlock!(
            g.ceres_problem,
            error_gps_3d_Tcw,
            e.gps3d,
            [e.v_wc.qwxyz, e.v_wc.xyz],
            3
        )

        # # set quaternion_parameterization
        Ceres.SetLocalParameterization!(
            g.ceres_problem,
            e.v_wc.qwxyz,
            Ceres.quaternion_parameterization
        )

    end
    @assert haskey(g.vertices, e.v_wc.id)



    push!(g.edges, e)
end
