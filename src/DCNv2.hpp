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
  int deformable_group_;
  int dilation_;
  int padding_;
  int stride_;

  cublasHandle_t cublas_handle_;
  bool ones_uploaded_;

  DCNv2Plugin();

public:
  void deserialize(void const * buffer, size_t length) noexcept
  {
    const char * data = reinterpret_cast<const char *>(buffer);
    read<decltype(deformable_group_)>(data, deformable_group_);
    read<decltype(dilation_)>(data, dilation_);
    read<decltype(padding_)>(data, padding_);
    read<decltype(stride_)>(data, stride_);
  }

  size_t getSerializationSize() const noexcept override
  {
    return sizeof(deformable_group_) +
           sizeof(dilation_) +
           sizeof(padding_) +
           sizeof(stride_);
  }

  void serialize(void * buffer) const noexcept override
  {
    char * data = reinterpret_cast<char *>(buffer);
    const char * start = data;
    write(data, deformable_group_);
    write(data, dilation_);
    write(data, padding_);
    write(data, stride_);
    assert(data == start + getSerializationSize());
  }

  DCNv2Plugin(
    int deformable_group,
    int dilation,
    int padding,
    int stride);

  DCNv2Plugin(void const * serialData, size_t serialLength)
  : DCNv2Plugin()
  {
    this->deserialize(serialData, serialLength);
    cublasCreate(&cublas_handle_);
  }

  virtual ~DCNv2Plugin();

  void destroy() noexcept override {}

  int32_t initialize() noexcept override
  {
    return 0;
  }

  void terminate() noexcept override {}

  void setPluginNamespace(const AsciiChar * pluginNamespace) noexcept override {}

  AsciiChar const * getPluginNamespace() const noexcept override {return "";}

  const nvinfer1::AsciiChar * getPluginType() const noexcept override {return "DCNv2";}

  const nvinfer1::AsciiChar * getPluginVersion() const noexcept override {return "1";}

  void getDcnOutputDimensions(
    const nvinfer1::PluginTensorDesc * inputs, int32_t & output_h,
    int32_t & output_w) const;

  int getNbOutputs() const noexcept override {return 1;}

  nvinfer1::DimsExprs getOutputDimensions(
    int outputIndex,
    const nvinfer1::DimsExprs * inputs,
    int nbInputs,
    nvinfer1::IExprBuilder & exprBuilder) noexcept override;

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
    namespace_ = pluginNamespace;
  }

  AsciiChar const * getPluginNamespace() const noexcept override
  {
    return namespace_.c_str();
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
  std::string namespace_;
};

} // namespace plugin
} // namespace nvinfer1
