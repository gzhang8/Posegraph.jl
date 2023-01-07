"""
switch_variable length one var. The optimizable variable
"""

struct VertexSwitchLinear{T<:Union{Int64, String}} <:Vertex
    # x ; y ; z ; qw; qx ; qy ; qz
    id::T
    fixed::Vector{Bool}
    switch_variable::Vector{Float64}
end


function VertexSwitchLinear(
    id::Union{Int64, String},
    estimate::Float64;
    fixed::Bool=false
)
    res = VertexSwitchLinear(id, [fixed], [estimate])
end


"""
se3_ab_info_weight: x, y, z, qw, qx, qy, qz, and column order 6x6 infomation
                    matrix, weight for switchable constraint
"""
struct EdgeSE3Switchable <: Edge
    v_s::VertexSwitchLinear
    v_wa::VertexSE3
    v_wb::VertexSE3
    se3_ab_info::Vector{Float64}
end


function EdgeSE3Switchable(
    v_s::VertexSwitchLinear,
    v_wa::VertexSE3,
    v_wb::VertexSE3,
    Tab::Matrix{Float64},
    information::Matrix{Float64};
    info_is_sqrt::Bool=false
)
    quat_ab = R2quat(Tab[1:3, 1:3]);
    xyzqwxzy = vcat(Tab[1:3, 4], quat_ab);

    EdgeSE3Switchable(
        v_s,
        v_wa,
        v_wb,
        xyzqwxzy,
        information,
        info_is_sqrt=info_is_sqrt
    )
end

function EdgeSE3Switchable(
    v_s::VertexSwitchLinear,
    v_wa::VertexSE3,
    v_wb::VertexSE3,
    xyzqwxzy::Vector{Float64},
    information::Matrix{Float64};
    info_is_sqrt::Bool=false
)

    # after this, information will be sqrt of information
    if !info_is_sqrt
        # C = LinearAlgebra.cholesky(information)
        # information = Matrix(C.U)

        information = sqrt(information)
    end

    se3_ab::Vector{Float64} = vcat(xyzqwxzy, information[:])
    EdgeSE3Switchable(v_s, v_wa, v_wb, se3_ab)
end



"""
Tab = inv(Twa) * Twb
r = Tab * inv(inv(Twa) * Twb)
ΔR = Rab * ̂inv(Rab) = Rab * inv(inv(Rwa) * Rwb)
p_ab_est = inv(Rwa)twb - inv(Rwa)twa by
   inv(Twa) * Twb = [ inv(Rwa)  -inv(Rwa)twa] * [ ... twb]
                    [   0             1     ]   [ 0     1]
quaternion vs matrix multiplication: which is faster?
A: use quaternion should be faster for tranfer point and inv rotation
The perfermance difference between ReferenceFrameRotations and Rotations
may due to @inline is widely used in ReferenceFrameRotations

Note: removing inv on inv(q_ab_estimated) will introduce other inv
so not try improve performance unless has a clear picture
"""
function edge_error_switchable_se3_Twc(userdata, parameters)
    p_ab = userdata[1:3] # x, y, z
    q_ab = Rotations.QuatRotation(userdata[4:7]...) # w, x, y, z TODO: check
    information_sqrt = reshape(userdata[8:8+35], (6, 6))
    p_a = parameters[1]
    q_a = Rotations.QuatRotation(parameters[2]...)

    p_b = parameters[3]
    q_b = Rotations.QuatRotation(parameters[4]...)

    switch_variable = parameters[5]


    # for unit quaternion, inv and conj is the same
    q_a_inverse = inv(q_a)

    q_ab_estimated = q_a_inverse * q_b

    delta_q = q_ab * inv(q_ab_estimated)
    # @show q_ab_estimated q_ab
    # res_q = Vector(2 * imag(delta_q))
    res_q = 2 * [delta_q.x, delta_q.y, delta_q.z]


    # p_ab_estimated = imag(q_a_inverse * (p_b - p_a) * q_a)
    p_ab_estimated = q_a_inverse * (p_b - p_a)


    delta_p = p_ab_estimated - p_ab

    residuals = [delta_p; res_q]

    # @show "a residdual" residuals

    residuals = information_sqrt * residuals .* switch_variable
    return residuals
end


"""
Tab = Taw * inv(Tbw)
Tab_est = Taw * inv(Tbw)

r = Tab * inv(Tab_est)
ΔR = Rab * ̂inv(Rab_est) = Rab * inv(Raw * inv(Rbw))

t_ab_est = -Raw * inv(Rbw)*tbw + taw
         = -Rab_est * tbw + taw
by
   Taw * inv(Tbw) = [ Raw taw]*[ inv(Rbw)  -inv(Rbw)tbw]
                    [ 0    1]  [   0             1     ]
quaternion vs matrix multiplication: which is faster?
A: use quaternion should be faster for tranfer point and inv rotation
The perfermance difference between ReferenceFrameRotations and Rotations
may due to @inline is widely used in ReferenceFrameRotations

Note: removing inv on inv(q_ab_estimated) will introduce other inv
so not try improve performance unless has a clear picture
"""
function edge_error_switchable_se3_Tcw(userdata, parameters)
    p_ab = userdata[1:3] # x, y, z
    q_ab = Rotations.QuatRotation(userdata[4:7]...) # w, x, y, z TODO: check
    information_sqrt = reshape(userdata[8:8+35], (6, 6))
    p_aw = parameters[1]
    q_aw = Rotations.QuatRotation(parameters[2]...)

    p_bw = parameters[3]
    q_bw = Rotations.QuatRotation(parameters[4]...)

    switch_variable = parameters[5]


    # for unit quaternion, inv and conj is the same

    q_wa = inv(q_aw)
    q_ba_estimated = q_bw * q_wa
    delta_q = q_ab * q_ba_estimated

    # @show q_ab_estimated q_ab
    # res_q = Vector(2 * imag(delta_q))
    res_q = 2 * [delta_q.x, delta_q.y, delta_q.z]


    # p_ab_estimated = imag(q_a_inverse * (p_b - p_a) * q_a)
    # p_ba_estimated = p_bw - q_bw *  (q_wa * p_aw)
    # delta_p = p_ba_estimated + p_ab

    p_ab_estimated = p_aw - q_aw *  (inv(q_bw) * p_bw)
    delta_p = p_ab_estimated - p_ab

    residuals = [delta_p; res_q]

    # @show "a residdual" residuals

    residuals = information_sqrt * residuals .* switch_variable
    return residuals
end


function add_edge!(g::Graph, e::EdgeSE3Switchable; is_Twc::Bool=true)
    if is_Twc
        Ceres.AddResidualBlock!(
            g.ceres_problem,
            edge_error_switchable_se3_Twc,
            e.se3_ab_info,
            [e.v_wa.xyz, e.v_wa.qwxyz, e.v_wb.xyz, e.v_wb.qwxyz, e.v_s.switch_variable],
            6
        )
    else
        Ceres.AddResidualBlock!(
            g.ceres_problem,
            edge_error_switchable_se3_Tcw,
            e.se3_ab_info,
            [e.v_wa.xyz, e.v_wa.qwxyz, e.v_wb.xyz, e.v_wb.qwxyz, e.v_s.switch_variable],
            6
        )
    end

    Ceres.SetLocalParameterization!(
        g.ceres_problem,
        e.v_wa.qwxyz,
        Ceres.quaternion_parameterization
    )

    Ceres.SetLocalParameterization!(
        g.ceres_problem,
        e.v_wb.qwxyz,
        Ceres.quaternion_parameterization
    )

    push!(g.edges, e)
end


struct EdgeSwitchPrior <: Edge
    v::VertexSwitchLinear
    prior_weight::Vector{Float64}
end

function EdgeSwitchPrior(
    v::VertexSwitchLinear;
    prior::Float64,
    weight::Float64)

    return EdgeSwitchPrior(v, [prior, sqrt(weight)])
end

function get_factor_estimate(v::VertexSwitchLinear)
    return v.switch_variable[1]
end


function edge_error_switchable_prior(userdata, parameters)
    prior = userdata[1]
    weight = userdata[2]
    est = parameters[1][1]
    return weight .* [prior - est]
end



function add_edge!(g::Graph, e::EdgeSwitchPrior)
    Ceres.AddResidualBlock!(
        g.ceres_problem,
        edge_error_switchable_prior,
        e.prior_weight,
        [e.v.switch_variable],
        1
    )

    Ceres.set_parameter_lower_bound!(
        g.ceres_problem,
        e.v.switch_variable,
        1,
        0.0
    )
    Ceres.set_parameter_upper_bound!(
        g.ceres_problem,
        e.v.switch_variable,
        1,
        1.0
    )

    push!(g.edges, e)
end
