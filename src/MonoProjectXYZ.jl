
"""
v_p3e: estimated 3d point vertex
obs_and_cam: [1:2] obseration 2d on images; 3: fx, 4: fy, 5: cx, 6: cy
           [7:10]: information_sqrt
"""
struct EdgeSE3ProjectXYZ{LT<:LossFunction} <: Edge
    v_cw::VertexSE3
    v_p3e::VertexPoint3d
    obs_and_cam::Vector{Float64}
    loss_function::LT
end


function EdgeSE3ProjectXYZ(
    v_cw::VertexSE3,
    v_p3e::VertexPoint3d,
    obs2d::Vector{<:Real},
    information::Matrix{<:AbstractFloat};
    fx::T,
    fy::T,
    cx::T,
    cy::T,
    loss_function::TL=EmptyLoss(),
    info_is_sqrt::Bool=false
) where {T<:Real, TL<:LossFunction}

    if !info_is_sqrt
        # C = LinearAlgebra.cholesky(information)
        # information = Matrix(C.L)
        information = sqrt(information)
    end
    obs_and_cam::Vector{Float64} = Float64[obs2d..., fx, fy, cx, cy, information...]
    EdgeSE3ProjectXYZ(v_cw, v_p3e, obs_and_cam, loss_function)
end

# https://github.com/ceres-solver/ceres-solver/blob/1.14.x/examples/bundle_adjuster.cc
"""
user_data: [1:2] obseration 2d on images; 3: fx, 4: fy, 5: cx, 6: cy
           [7:10]: information_sqrt

parameters[1]: t_cw
parameters[2]: q_cw as vector: w, x, y, z
parameters[3]: p3d_w
"""
function reprojection_error_wo_undistort(
    user_data,
    parameters
)
    observation2d = user_data[1:2]
    fx, fy, cx, cy = user_data[3:6]

    information_sqrt = reshape(user_data[7:10], (2, 2))

    t_cw = parameters[1] # x, y, z
    q_cw = Rotations.QuatRotation(parameters[2]...) # w, x, y, z
    p3d_w = parameters[3] # xyz

    p3d_c = q_cw * p3d_w
    p3d_c += t_cw

    x, y, z = p3d_c

    invz = 1/z
    predicted_x = fx * invz * x + cx
    predicted_y = fy * invz * y + cy

    residuals = [predicted_x; predicted_y] - reshape(observation2d, (2, 1))
    residuals = information_sqrt * residuals
    return residuals
end


function add_edge!(g::Graph, e::EdgeSE3ProjectXYZ)

    # parameters[1]: t_cw
    # parameters[2]: q_cw as vector: w, x, y, z
    # parameters[3]: p3d_w
    Ceres.AddResidualBlock!(
        g.ceres_problem,
        reprojection_error_wo_undistort,
        e.obs_and_cam,
        [e.v_cw.xyz, e.v_cw.qwxyz, e.v_p3e.xyz],
        2,
        loss_function=e.loss_function
    )

    Ceres.SetLocalParameterization!(
        g.ceres_problem,
        e.v_cw.qwxyz,
        Ceres.quaternion_parameterization
    )

    push!(g.edges, e)
end


function get_error(e::EdgeSE3ProjectXYZ)
    parameters = [e.v_cw.xyz, e.v_cw.qwxyz, e.v_p3e.xyz]
    user_data = e.obs_and_cam
    error = reprojection_error_wo_undistort(
        user_data,
        parameters
    )
    return error
end

function get_information(e::EdgeSE3ProjectXYZ)
    reshape(e.obs_and_cam[7:10], (2, 2))
end

function getχ²(e::EdgeSE3ProjectXYZ)
    error = get_error(e)
    # information = get_information(e)
    # χ² = error' * (information * error)
    χ² = error' * error # TODO GZ I think this error has included information unlike g2o
    return χ²[1]
end


function is_depth_positive(e::EdgeSE3ProjectXYZ) 
    # [e.v_cw.xyz, e.v_cw.qwxyz, e.v_p3e.xyz],
    t_cw = e.v_cw.xyz # x, y, z
    q_cw = Rotations.QuatRotation(e.v_cw.qwxyz...) # w, x, y, z
    p3d_w = e.v_p3e.xyz # xyz

    p3d_c = q_cw * p3d_w
    p3d_c += t_cw

    z = p3d_c[3]
    return z > 0
end