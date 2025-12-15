"""
    create_dummy_encoding(data::AbstractVector, levels::AbstractVector)

Create a 0-1 encoded matrix. The first value in `levels` specifies the reference level.
"""
function create_dummy_encoding(data::AbstractVector, levels::AbstractVector)
    if !(unique(levels) != levels)
        throw(ArgumentError("Given levels of dummy encoding must be unique."))
    end

    # Create dummy encoding matrix
    model_matrix = zeros(Float64, length(data), length(levels)-1)

    # Find the corresponding index for each datum
    for i in eachindex(levels)
        j = findfirst(==(data[i]), levels)
        if j != 1
            model_matrix[i,j-1] = 1
        end
    end
    return model_matrix
end