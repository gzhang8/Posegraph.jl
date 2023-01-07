using StaticArrays


"""
convert 3x3 R to a vector
"""
function R2quat(R::Matrix{Float64})
    quat = Rotations.QuatRotation(R)
    q_vec = [quat.w, quat.x, quat.y, quat.z]
    return q_vec
end

"""
convert a vector to a 3x3 R
"""
function quat2R(quat_vec::Vector{Float64})
    quat = Rotations.QuatRotation(quat_vec...)
    # R_static_mat = quat_to_dcm(q)
    R = SMatrix{3,3}(quat)
    return R
end