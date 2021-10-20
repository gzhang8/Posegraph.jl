"""
v_p3e: estimated 3d point vertex
obs_and_cam:  [1:3] stereo obseration on images x, y, other_x;
              4: fx, 5: fy, 6: cx, 7: cy, 8: bf
              [9:17]: information_sqrt
"""
struct EdgeStereoSE3ProjectXYZ{LT<:LossFunction} <: Edge
    v_cw::VertexSE3
    v_p3e::VertexPoint3d
    obs_and_cam_bf::Vector{Float64}
    loss_function::LT
end


function EdgeStereoSE3ProjectXYZ(
    v_cw::VertexSE3,
    v_p3e::VertexPoint3d,
    obs_stereo::Vector{<:Real},
    information::Matrix{<:AbstractFloat};
    fx::T,
    fy::T,
    cx::T,
    cy::T,
    bf::T,
    loss_function::TL=EmptyLoss(),
    info_is_sqrt::Bool=false
) where {T<:Real, TL<:LossFunction}

    if !info_is_sqrt
        # C = LinearAlgebra.cholesky(information)
        # information = Matrix(C.L)
        information = sqrt(information)
    end
    obs_and_cam::Vector{Float64} = Float64[obs_stereo..., fx, fy, cx, cy, bf,
                                           information...]
    EdgeStereoSE3ProjectXYZ(v_cw, v_p3e, obs_and_cam, loss_function)
end
# https://github.com/ceres-solver/ceres-solver/blob/1.14.x/examples/bundle_adjuster.cc
"""
user_data: [1:3] stereo obseration on images x, y, other_x;
           4: fx, 5: fy, 6: cx, 7: cy, 8: bf
           [9:17]: information_sqrt

parameters[1]: t_cw
parameters[2]: q_cw as vector: w, x, y, z
parameters[3]: p3d_w
"""
function stereo_reprojection_error_wo_undistort(
    user_data,
    parameters
)
    obs_stereo = user_data[1:3]
    fx, fy, cx, cy, bf = user_data[4:8]

    information_sqrt = reshape(user_data[9:17], (3, 3))

    t_cw = parameters[1] # x, y, z
    q_cw = Rotations.Quat(parameters[2]...) # w, x, y, z
    p3d_w = parameters[3] # xyz

    p3d_c = q_cw * p3d_w
    p3d_c += t_cw

    x, y, z = p3d_c

    invz = 1/z
    predicted_x = fx * invz * x + cx
    predicted_y = fy * invz * y + cy
    predicted_other_x = predicted_x - bf*invz;
     # res[0] = trans_xyz[0]*invz*fx + cx;
     # res[1] = trans_xyz[1]*invz*fy + cy;
     # res[2] = res[0] - bf*invz;


    residuals = [predicted_x; predicted_y; predicted_other_x] - reshape(obs_stereo, (3, 1))
    residuals = information_sqrt * residuals
    return residuals
end


function add_edge!(g::Graph, e::EdgeStereoSE3ProjectXYZ)

    # parameters[1]: t_cw
    # parameters[2]: q_cw as vector: w, x, y, z
    # parameters[3]: p3d_w
    Ceres.AddResidualBlock!(
        g.ceres_problem,
        stereo_reprojection_error_wo_undistort,
        e.obs_and_cam_bf,
        [e.v_cw.xyz, e.v_cw.qwxyz, e.v_p3e.xyz],
        3,
        loss_function=e.loss_function
    )

    Ceres.SetLocalParameterization!(
        g.ceres_problem,
        e.v_cw.qwxyz,
        Ceres.quaternion_parameterization
    )

    push!(g.edges, e)
end

function get_error(e::EdgeStereoSE3ProjectXYZ)
    parameters = [e.v_cw.xyz, e.v_cw.qwxyz, e.v_p3e.xyz]
    user_data = e.obs_and_cam_bf
    error = stereo_reprojection_error_wo_undistort(
        user_data,
        parameters
    )
    return error
end

function get_information(e::EdgeStereoSE3ProjectXYZ)
    information = reshape(e.obs_and_cam_bf[9:17], (3, 3))
    return information
end

function getχ²(e::EdgeStereoSE3ProjectXYZ)
    error = get_error(e)
    information = get_information(e)
    χ² = error' * (information * error)
    return χ²[1]
end
