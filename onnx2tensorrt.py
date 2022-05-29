import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)

builder = trt.Builder(logger)

network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

parser = trt.OnnxParser(network, logger)

success = parser.parse_from_file('model.onnx')
# for idx in range(parser.num_errors):
#     print(parser.get_error(idx))

if not success:
    pass # Error handling code here

config = builder.create_builder_config()
#config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20) # 1 MiB
config.max_workspace_size = 1 << 31

profile = builder.create_optimization_profile()  # 动态输入时候需要 分别为最小输入、常规输入、最大输入
# 有几个输入就要写几个profile.set_shape 名字和转onnx的时候要对应
# tensorrt6以后的版本是支持动态输入的，需要给每个动态输入绑定一个profile，用于指定最小值，常规值和最大值，如果超出这个范围会报异常。
profile.set_shape("input_ids", (1, 1), (1, 20), (1, 300))
profile.set_shape("token_type_ids", (1, 1), (1, 20), (1, 300))
config.add_optimization_profile(profile)

serialized_engine = builder.build_serialized_network(network, config)
with open("sample4.engine", "wb") as f:
    f.write(serialized_engine)




