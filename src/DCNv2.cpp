#include "DCNv2.hpp"
#include "NvInfer.h"
#include "dcn_v2_im2col_cuda.h"
#include <iostream>
#include <assert.h>
#include <vector>

using namespace nvinfer1; // NOLINT
using nvinfer1::plugin::DCNv2Plugin;
using nvinfer1::plugin::DCNv2PluginCreator;

#define CHECK_CUDA(call) do { \
    cudaError_t status = call; \
    if (status != cudaSuccess) { \
      return status; \
    } \
} while (0)

void getDCNWeightDims(
  const PluginTensorDesc & weight, int32_t & out_channel,
  int32_t & in_channel, int32_t & kernel_size)
{
  auto weight_dims = weight.dims;
  // weight input dimensions: out_channel X in_channel // groups X kernel size X kernel size
  assert(weight_dims.nbDims == 4);
  out_channel = weight_dims.d[0];
  in_channel = weight_dims.d[1];
  kernel_size = weight_dims.d[2];
  // TODO(ZeilingerM) enable h different of w?
  assert(kernel_size == weight_dims.d[3]);
}

DCNv2Plugin::DCNv2Plugin()
: deformable_group_(1), dilation_(1), padding_(1), stride_(1), ones_uploaded_(false)
{
  cublasCreate(&cublas_handle_);
}

DCNv2Plugin::DCNv2Plugin(
  int deformable_group,
  int dilation,
  int padding,
  int stride)
: DCNv2Plugin()
{
  deformable_group_ = deformable_group;
  dilation_ = dilation;
  padding_ = padding;
  stride_ = stride;
}

DCNv2Plugin::~DCNv2Plugin()
{
  cublasDestroy(cublas_handle_);
}

void DCNv2Plugin::configurePlugin(
  const nvinfer1::DynamicPluginTensorDesc * in, int nbInputs,
  const nvinfer1::DynamicPluginTensorDesc * out, int nbOutputs) noexcept
{
  assert(nbInputs == 5);
  assert(nbOutputs == 1);
}

bool DCNv2Plugin::supportsFormatCombination(
  int pos, const nvinfer1::PluginTensorDesc * desc,
  int nbInputs, int nbOutputs) noexcept
{
  assert(nbInputs == 5);
  assert(nbOutputs == 1);
  assert(pos < (nbInputs + nbOutputs));
  if (pos <= 2) {
    // inputs
    return (desc[pos].type == nvinfer1::DataType::kFLOAT) &&
           (desc[pos].format == nvinfer1::TensorFormat::kCHW32);
  }
  if (pos == 3) {
    // weight
    return (desc[pos].type == nvinfer1::DataType::kFLOAT) &&
           (desc[pos].format == nvinfer1::TensorFormat::kCHW32);
  }
  if (pos == 4) {
    // bias
    return (desc[pos].type == nvinfer1::DataType::kFLOAT) &&
           (desc[pos].format == nvinfer1::TensorFormat::kLINEAR);
  }
  if (pos == 5) {
    // output
    return (desc[pos].type == nvinfer1::DataType::kFLOAT) &&
           (desc[pos].format == nvinfer1::TensorFormat::kCHW32);
  }
  return false;
}

nvinfer1::DimsExprs DCNv2Plugin::getOutputDimensions(
  int outputIndex,
  const nvinfer1::DimsExprs * inputs,
  int nbInputs,
  nvinfer1::IExprBuilder & exprBuilder) noexcept
{
  assert(outputIndex == 0);
  assert(nbInputs == 5);

  // weight input dimensions: out_channel X in_channel // groups X kernel size X kernel size
  auto weight_dims = inputs[3];
  assert(weight_dims.nbDims == 4);
  auto out_channel = weight_dims.d[0]->getConstantValue();
  auto kernel_size = weight_dims.d[2]->getConstantValue();

  nvinfer1::DimsExprs output(inputs[0]);
  auto input_h = output.d[2]->getConstantValue();
  auto input_w = output.d[3]->getConstantValue();
  auto output_h = (input_h + 2 * padding_ - (dilation_ * (kernel_size - 1) + 1)) / stride_ + 1;
  auto output_w = (input_w + 2 * padding_ - (dilation_ * (kernel_size - 1) + 1)) / stride_ + 1;
  output.d[1] = exprBuilder.constant(out_channel);
  output.d[2] = exprBuilder.constant(output_h);
  output.d[3] = exprBuilder.constant(output_w);
  return output;
}

void DCNv2Plugin::getDcnOutputDimensions(
  const nvinfer1::PluginTensorDesc * inputDesc,
  int32_t & output_h, int32_t & output_w) const
{
  auto input_dims = inputDesc[0].dims;

  int32_t kernel_size, _ignored;
  getDCNWeightDims(inputDesc[3], _ignored, _ignored, kernel_size);

  output_h = (input_dims.d[2] + 2 * padding_ - (dilation_ * (kernel_size - 1) + 1)) / stride_ + 1;
  output_w = (input_dims.d[3] + 2 * padding_ - (dilation_ * (kernel_size - 1) + 1)) / stride_ + 1;
}

size_t DCNv2Plugin::getWorkspaceSize(
  const nvinfer1::PluginTensorDesc * inputs, int nbInputs,
  const nvinfer1::PluginTensorDesc * outputs,
  int nbOutputs) const noexcept
{
  assert(nbInputs == 5);
  assert(nbOutputs == 1);

  int32_t _ignored, in_channel, kernel_size, output_h, output_w;
  getDCNWeightDims(inputs[3], _ignored, in_channel, kernel_size);
  getDcnOutputDimensions(inputs, output_h, output_w);

  size_t ones_size = output_h * output_w * sizeof(float);
  size_t columns_size = in_channel * kernel_size * kernel_size * ones_size;
  return ones_size + columns_size;
}

int DCNv2Plugin::enqueue(
  const nvinfer1::PluginTensorDesc * inputDesc,
  const nvinfer1::PluginTensorDesc * outputDesc,
  const void * const * inputs, void * const * outputs,
  void * workspace, cudaStream_t stream) noexcept
{
  // initialize inputs

  const float * input = static_cast<const float *>(inputs[0]);
  const float * offset = static_cast<const float *>(inputs[1]);
  const float * mask = static_cast<const float *>(inputs[2]);
  const float * weight = static_cast<const float *>(inputs[3]);
  const float * bias = static_cast<const float *>(inputs[4]);

  assert(inputDesc[0].dims.nbDims == 4);
  int h = inputDesc[0].dims.d[2];
  int w = inputDesc[0].dims.d[3];

  // initialize output

  float * output = static_cast<float *>(outputs[0]);
  int32_t out_channel, in_channel, kernel_size, output_h, output_w;
  getDCNWeightDims(inputDesc[3], out_channel, in_channel, kernel_size);
  getDcnOutputDimensions(inputDesc, output_h, output_w);

  // initialize workspace

  size_t ones_size = output_h * output_w * sizeof(float);
  if (!ones_uploaded_) {
    std::vector<float> ones_h;
    ones_h.resize(output_h * output_w, 1.0);
    CHECK_CUDA(cudaMemcpy(workspace, ones_h.data(), ones_size, cudaMemcpyHostToDevice));
    ones_uploaded_ = true;
  }
  auto ones = static_cast<float *>(workspace);
  auto columns = static_cast<float *>(workspace) + (output_h * output_w);

  // run DCN
  float alpha, beta;
  int m, n, k;

  m = out_channel;
  n = output_h * output_w;
  k = 1;
  alpha = 1.0;
  beta = 0.0;
  /// output  nxm
  /// ones    1xn  T ->> nx1
  /// bias    1xm
  /// ones x bias = nxm
  //  add bias
  cublasSgemm(
    cublas_handle_,
    CUBLAS_OP_T, CUBLAS_OP_N,
    n, m, k, &alpha,
    ones, k,
    bias, k, &beta,
    output, n);
  // im2col (offset and mask)
  modulated_deformable_im2col_cuda(
    stream, input, offset, mask,
    1, in_channel, h, w,
    output_h, output_w, kernel_size, kernel_size,
    padding_, padding_, stride_, stride_, dilation_, dilation_,
    deformable_group_, columns);

  k = in_channel * kernel_size * kernel_size;
  alpha = 1.0;
  beta = 1.0;

  // im2col conv
  cublasSgemm(
    cublas_handle_,
    CUBLAS_OP_N, CUBLAS_OP_N,
    n, m, k, &alpha,
    columns, n,
    weight, k,
    &beta,
    output, n);
  return 0;
}

nvinfer1::DataType DCNv2Plugin::getOutputDataType(
  int index, const nvinfer1::DataType * inputTypes,
  int nbInputs) const noexcept
{
  assert(index == 0);
  assert(nbInputs == 5);
  assert(inputTypes[0] == nvinfer1::DataType::kFLOAT);
  return inputTypes[0];
}

nvinfer1::IPluginV2DynamicExt * DCNv2Plugin::clone() const noexcept
{
  return new DCNv2Plugin(deformable_group_, dilation_, padding_, stride_);
}

PluginFieldCollection DCNv2PluginCreator::mFC{};
std::vector<PluginField> DCNv2PluginCreator::mPluginAttributes;

DCNv2PluginCreator::DCNv2PluginCreator()
{
  mPluginAttributes.emplace_back(
    PluginField(
      "deformable_group", nullptr, PluginFieldType::kINT32,
      1));
  mPluginAttributes.emplace_back(PluginField("dilation", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(PluginField("padding", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(PluginField("stride", nullptr, PluginFieldType::kINT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char * DCNv2PluginCreator::getPluginName() const noexcept
{
  return "DCNv2";
}

const char * DCNv2PluginCreator::getPluginVersion() const noexcept
{
  return "1";
}

const PluginFieldCollection * DCNv2PluginCreator::getFieldNames() noexcept
{
  return &mFC;
}

IPluginV2DynamicExt * DCNv2PluginCreator::createPlugin(
  const char * name,
  const nvinfer1::PluginFieldCollection * fc) noexcept
{
  int deformable_group, padding, stride, dilation;
  const PluginField * fields = fc->fields;
  for (int i = 0; i < fc->nbFields; ++i) {
    const char * attrName = fields[i].name;
    if (!strcmp(attrName, "deformable_group")) {
      assert(fields[i].type == PluginFieldType::kINT32);
      deformable_group = *(static_cast<const int *>(fields[i].data));
    } else if (!strcmp(attrName, "dilation")) {
      assert(fields[i].type == PluginFieldType::kINT32);
      dilation = *(static_cast<const int *>(fields[i].data));
    } else if (!strcmp(attrName, "stride")) {
      assert(fields[i].type == PluginFieldType::kINT32);
      stride = *(static_cast<const int *>(fields[i].data));
    } else if (!strcmp(attrName, "padding")) {
      assert(fields[i].type == PluginFieldType::kINT32);
      padding = *(static_cast<const int *>(fields[i].data));
    }
  }

  DCNv2Plugin * obj = new DCNv2Plugin(deformable_group, dilation, padding, stride);
  obj->setPluginNamespace(namespace_.c_str());
  return obj;
}

IPluginV2DynamicExt * DCNv2PluginCreator::deserializePlugin(
  const char * name,
  const void * serialData,
  size_t serialLength) noexcept
{
  DCNv2Plugin * obj = new DCNv2Plugin(serialData, serialLength);
  obj->setPluginNamespace(namespace_.c_str());
  return obj;
}

REGISTER_TENSORRT_PLUGIN(DCNv2PluginCreator);
