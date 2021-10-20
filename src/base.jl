"""
each Edge has to implement get_id_set function which returns a set of vertex ids
"""
abstract type Edge end



abstract type Vertex end


mutable struct Graph
    ceres_problem::Ceres.Problem
    vertices::Dict{Union{Int64, String}, Vertex}
    edges::Vector{Edge}
end

function Graph()
    ceres_problem = Ceres.Problem()
    vertices = Dict{Union{Int64, String}, Vertex}()
    edges = Vector{Edge}(undef, 0)
    Graph(ceres_problem, vertices, edges)
end

function add_vertex!(g::Graph, v::T) where T<:Vertex
    g.vertices[v.id] = v
end

# function add_edge_base!(g::Graph, e::T) where T<:Edge
#     push!(g.edges, e)
#     e_ids = get_id_set(e)
#     @assert !haskey(g.edge_key_dict, e_ids)
#     g.edge_key_dict[e_ids] = e
# end

function solve!(g::Graph; max_iter_num::Int64=100, solver_type::Int64=1)
    Ceres.solve(
        g.ceres_problem,
        max_iter_num=max_iter_num,
        solver_type=solver_type
    )
end
