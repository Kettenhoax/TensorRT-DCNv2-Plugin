#pragma once

#include <vector>
#include <string>
#include "NvInferPlugin.h"
#include "NvInferRuntime.h"
#include <cudnn.h>
#include <cublas_v2.h>
#include <assert.h>

namespace nvinfer1
{
namespace plugin
{

template<typename T>
void read(const char * & buffer, T & val)
{
  val = *reinterpret_cast<const T *>(buffer);
  buffer += sizeof(T);
}

template<typename T>
void write(char * & buffer, const T & val)
{
  *reinterpret_cast<T *>(buffer) = val;
  buffer += sizeof(T);
}

class DCNv2Plugin final : public nvinfer1::IPluginV2DynamicExt
{
private:
  int _in_channel;
  int _out_channel;
  int _kernel_H;
  int _kernel_W;
  int _deformable_group;
  int _dilation;
  int _groups; // not use
  int _padding;
  int _stride;
  int _output_height;
  int _output_width;
  std::vector<float> _h_weight;
  std::vector<float> _h_bias;
  float * _d_weight;
  float * _d_bias;
  float * _d_ones;
  float * _d_columns;
  cublasHandle_t _cublas_handle;
  const char * _plugin_namespace;

  bool _initialized;

public:
  void deserialize(void const * buffer, size_t length) noexcept
  {
    const char * data = reinterpret_cast<const char *>(buffer);

    read<decltype(_in_channel)>(data, _in_channel);
    read<decltype(_out_channel)>(data, _out_channel);
    read<decltype(_kernel_H)>(data, _kernel_H);
    read<decltype(_kernel_W)>(data, _kernel_W);
    read<decltype(_deformable_group)>(data, _deformable_group);
    read<decltype(_dilation)>(data, _dilation);
    read<decltype(_groups)>(data, _groups);
    read<decltype(_padding)>(data, _padding);
    read<decltype(_stride)>(data, _stride);

    size_t n_weight;
    read<size_t>(data, n_weight);
    _h_weight.resize(n_weight);
    for (size_t i = 0; i < n_weight; i++) {
      read<decltype(_h_weight)::value_type>(data, _h_weight[i]);
    }

    size_t n_bias;
    read<size_t>(data, n_bias);
    _h_bias.resize(n_bias);
    for (size_t i = 0; i < n_bias; i++) {
      read<decltype(_h_bias)::value_type>(data, _h_bias[i]);
    }

    read<decltype(_output_height)>(data, _output_height);
    read<decltype(_output_height)>(data, _output_height);
  }

  size_t getSerializationSize() const noexcept override
  {
    return sizeof(_in_channel) +
           sizeof(_out_channel) +
           sizeof(_kernel_H) +
           sizeof(_kernel_W) +
           sizeof(_deformable_group) +
           sizeof(_dilation) +
           sizeof(_groups) +
           sizeof(_padding) +
           sizeof(_stride) +
           _h_weight.size() * sizeof(decltype(_h_weight)::value_type) +
           _h_bias.size() * sizeof(decltype(_h_bias)::value_type) +
           sizeof(_output_height) +
           sizeof(_output_width)
    ;
  }

  void serialize(void * buffer) const noexcept override
  {
    char * data = reinterpret_cast<char *>(buffer);
    const char * start = data;
    write(data, _in_channel);
    write(data, _out_channel);
    write(data, _kernel_H);
    write(data, _kernel_W);
    write(data, _deformable_group);
    write(data, _dilation);
    write(data, _groups);
    write(data, _padding);
    write(data, _stride);
    write(data, _h_weight.size());
    for (size_t i = 0; i < _h_weight.size(); i++) {
      write(data, _h_weight[i]);
    }
    write(data, _h_bias.size());
    for (size_t i = 0; i < _h_bias.size(); i++) {
      write(data, _h_bias[i]);
    }
    write(data, _output_height);
    write(data, _output_width);
    assert(data == start + getSerializationSize());
  }

  DCNv2Plugin(
    int in_channel,
    int out_channel,
    int kernel_H,
    int kernel_W,
    int deformable_group,
    int dilation,
    int groups,
    int padding,
    int stride,
    nvinfer1::Weights const & weight,
    nvinfer1::Weights const & bias);

  DCNv2Plugin(
    int in_channel,
    int out_channel,
    int kernel_H,
    int kernel_W,
    int deformable_group,
    int dilation,
    int groups,
    int padding,
    int stride,
    const std::vector<float> & weight,
    const std::vector<float> & bias);

  DCNv2Plugin(void const * serialData, size_t serialLength)
  : _initialized(false)
  {
    this->deserialize(serialData, serialLength);
    cublasCreate(&_cublas_handle);
  }

  DCNv2Plugin() = delete;

  const nvinfer1::AsciiChar * getPluginType() const noexcept override {return "DCNv2";}

  const nvinfer1::AsciiChar * getPluginVersion() const noexcept override {return "1";}

  void destroy() noexcept override;

  int getNbOutputs() const noexcept override {return 1;}

  nvinfer1::DimsExprs getOutputDimensions(
    int outputIndex,
    const nvinfer1::DimsExprs * inputs,
    int nbInputs,
    nvinfer1::IExprBuilder & exprBuilder) noexcept override;

  int32_t initialize() noexcept override;

  void terminate() noexcept override;

  int enqueue(
    const nvinfer1::PluginTensorDesc * inputDesc,
    const nvinfer1::PluginTensorDesc * outputDesc,
    const void * const * inputs, void * const * outputs,
    void * workspace, cudaStream_t stream) noexcept override;

  size_t getWorkspaceSize(
    const nvinfer1::PluginTensorDesc * inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc * outputs, int nbOutputs) const noexcept override;

  bool supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc * inOut, int nbInputs,
    int nbOutputs) noexcept override;

  nvinfer1::IPluginV2DynamicExt * clone() const noexcept override;

  void setPluginNamespace(const AsciiChar * pluginNamespace) noexcept override
  {
    _plugin_namespace = pluginNamespace;
  }

  AsciiChar const * getPluginNamespace() const noexcept override {return _plugin_namespace;}

  nvinfer1::DataType getOutputDataType(
    int index, const nvinfer1::DataType * inputTypes,
    int nbInputs) const noexcept override;

  void attachToContext(
    cudnnContext * cudnn, cublasContext * cublas,
    nvinfer1::IGpuAllocator * allocator) noexcept override {}

  void detachFromContext() noexcept override {}

  void configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc * in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc * out, int nbOutputs) noexcept override;
  ~DCNv2Plugin();
};

class DCNv2PluginCreator : public IPluginCreator
{
public:
  DCNv2PluginCreator();

  ~DCNv2PluginCreator() override = default;

  const nvinfer1::AsciiChar * getPluginName() const noexcept override;

  const nvinfer1::AsciiChar * getPluginVersion() const noexcept override;

  const PluginFieldCollection * getFieldNames() noexcept override;

  void setPluginNamespace(const AsciiChar * pluginNamespace) noexcept override
  {
    mNamespace = pluginNamespace;
  }

  AsciiChar const * getPluginNamespace() const noexcept override
  {
    return mNamespace.c_str();
  }

  IPluginV2DynamicExt * createPlugin(
    const char * name,
    const nvinfer1::PluginFieldCollection * fc) noexcept override;

  IPluginV2DynamicExt * deserializePlugin(
    const char * name, const void * serialData,
    size_t serialLength) noexcept override;

private:
  static PluginFieldCollection mFC;
  static std::vector<PluginField> mPluginAttributes;
  std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1
