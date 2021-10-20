

struct VertexPoint3d{T<:Union{Int64, String}} <:Vertex
    id::T
    fixed::Vector{Bool}
    xyz::Vector{Float64}
end


function VertexPoint3d(
    id::Union{Int64, String},
    xyz::Vector{Float64};
    fixed::Bool=false
)
    VertexPoint3d(id, [fixed], xyz)
end
