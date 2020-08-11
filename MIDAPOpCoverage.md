Only integer operation

- &#x1F49A; The standard API (Based on ONNX) can map to a custom operator.
- &#x1F49B; The solution is not perfect/finished, for example, the API can be decomposed to combination of custom operators. (or to be supported soon - TBS)
- &#x1F494; Hard to find a solution with existing MIDAP architecture.

| Operator | Test Coverage | From Existent Models (Caffe2) | Custom Operator |
|---|:--:|:---:|:---:|
|Abs||No|&#x1F49B; Activation module extension is required|
|Add|Yes|OK|&#x1F49A;OK|
|ArgMax|||&#x1F494;No op|
|ArgMin|||&#x1F494;No op|
|AveragePool|Yes|OK|&#x1F49A;OK|
|BatchNormalization|Yes|OK|&#x1F49B; Merged into preceding Convolution|
|Ceil|||&#x1F494;No op - integer only|
|Clip|||&#x1F49B; ReLu6 only; Activation module extension is required|
|Concat|Yes|OK|&#x1F49A;OK|
|Conv|Yes|OK|&#x1F49A;OK|
|ConvTranspose|Yes||&#x1F49B;Decomposed to Zeropadding and Convolution|
|Div|||&#x1F49B; TBS|
|Dropout|||&#x1F494; No Idea|
|Equal|||&#x1F494;|
|Exp|||&#x1F494;|
|Flatten|Yes||&#x1F49B; Via Tensor Virtualization|
|Floor|||&#x1F494;No op - integer only|
|GRU|||&#x1F49B; TBS|
|GlobalAveragePool|Yes|OK|&#x1F49A;OK|
|GlobalLpPool|||&#x1F494;No op|
|GlobalMaxPool|Yes|OK|&#x1F49A;OK|
|InstanceNormalization|||&#x1F494;|
|LRN|||&#x1F494;|
|LSTM|||&#x1F49B; TBS|
|LeakyRelu|Yes|OK|&#x1F49A;OK|
|Log|||&#x1F49B; Activation module extentsion is required|
|MatMul|Yes|OK|&#x1F49A;OK (Gemm)|
|Max|||&#x1F494;|
|MaxPool|Yes|OK|&#x1F49A;OK|
|MaxRoiPool|||&#x1F494;No op|
|Mean|||&#x1F494;No op|
|Min|||&#x1F494;|
|Mul|||&#x1F49B; TBS|
|Neg|||&#x1F49B; TBS|
|PRelu|||&#x1F49B; TBS(Activation module extension is required)|
|Pad||OK|&#x1F49B; Only Zero-padding in preprocessing step|
|Pow||OK|&#x1F49B; TBS(Activation module extension is required)|
|RNN|||&#x1F49B; TBS|
|ReduceMax|||&#x1F494;No op|
|ReduceMean|||&#x1F494;No op|
|ReduceMin|||&#x1F494;No op|
|ReduceProd|||&#x1F494;No op|
|ReduceSum|||&#x1F494;No op|
|ReduceSumSquare|||&#x1F494;No op|
|Relu|Yes|OK|&#x1F49A;OK|
|Reshape|||&#x1F49B;Tensor virtualization, patterned reshaping is only supported|
|Sigmoid|Yes|OK|&#x1F49A;OK|
|Slice|||&#x1F494;|
|Softmax|Yes|OK|&#x1F494;Axis and dim has different semantics|
|Split|||&#x1F494; Multiple output is not supported in MIDAP|
|Sqrt|||&#x1F49B; Activation Module extension is required|
|Squeeze|||&#x1F49B; TBS (Activation Module Extension)|
|Sub|||&#x1F49A; Not tested, but it is equal to Add|
|SubpixelConv|||&#x1F49A; Tensor Virtualization|
|Sum|Yes|OK|&#x1F49A; OK|
|Tanh|||&#x1F49B; TBS (Activation Module)|
|Transpose|||&#x1F49B; Tensor Virtualization may be able to support this operator|
|Upsample|||&#x1F494; Only Nearest Neighbor|

## Skipped Operators
| Operator | Test Coverage | From Existent Models (Caffe2) | Custom Operator |
|---|:--:|:---:|:---:|
|And|||
|Cast|||
|Constant|||
|DepthToSpace|||
|Elu|||
|Gather|||
|Gemm|||
|Greater|||
|HardSigmoid|||
|Hardmax|||
|Less|||
|LogSoftmax|||
|LpNormalization|||
|LpPool|||
|Not|||
|Or|||
|RandomNormal|||
|RandomNormalLike|||
|RandomUniform|||
|RandomUniformLike|||
|Reciprocal|||
|ReduceL1|||
|ReduceL2|||
|ReduceLogSum|||
|ReduceLogSumExp|||
|Selu|||
|Softplus|||
|Softsign|||
|SpaceToDepth|||
|Tile|||
|Xor|||
|experimental ATen|||
|experimental Affine|||
|experimental ConstantFill|||
|experimental Crop|||
|experimental FC|||
|experimental GRUUnit|||
|experimental GivenTensorFill|||
|experimental Identity|||
|experimental ImageScaler|||
|experimental MeanVarianceNormalization|||
|experimental ParametricSoftplus|||
|experimental Scale|||
|experimental ScaledTanh|||
|experimental ThresholdedRelu|||

