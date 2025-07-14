# ONNXRunner.jl
module ONNXRunner

export ONNXModel, refine, score

using ONNXRunTime: load_inference
using CUDA, cuDNN

function ONNXModel(model_path::String)
    return load_inference(model_path, execution_provider=:cuda)
end

function refine(onnx_model, input1, input2)
    output = onnx_model(Dict(["input1" => input1, "input2" => input2]))
    return output["output1"], output["output2"]
end

function score(onnx_model, input1, input2)
    output = onnx_model(Dict(["input1" => input1, "input2" => input2]))
    return output["output1"]
end

end # module ONNXRunner
