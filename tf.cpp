

#include <fstream>
#include <utility>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

static Status ReadEntireFile(tensorflow::Env* env, const string& filename,
                             Tensor* output) {
  tensorflow::uint64 file_size = 0;
  TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

  string contents;
  contents.resize(file_size);

  std::unique_ptr<tensorflow::RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

  tensorflow::StringPiece data;
  TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
  if (data.size() != file_size) {
    return tensorflow::errors::DataLoss("Truncated read of '", filename,
                                        "' expected ", file_size, " got ",
                                        data.size());
  }
  output->scalar<string>()() = data.ToString();

  return Status::OK();
}

Status ReadTensorFromImageFile(const string& file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor>* out_tensors) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string input_name = "file_reader";
  string output_name = "normalized";

  // read file_name into a tensor named input
  Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
  TF_RETURN_IF_ERROR(
      ReadEntireFile(tensorflow::Env::Default(), file_name, &input));

  // use a placeholder to read input data
  auto file_reader =
      Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
      {"input", input},
  };

  // Now try to figure out what kind of file it is and decode it.
  const int wanted_channels = 1;
  tensorflow::Output image_reader;
  if (tensorflow::StringPiece(file_name).ends_with(".png")) {
    image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
                             DecodePng::Channels(wanted_channels));
  } else if (tensorflow::StringPiece(file_name).ends_with(".gif")) {
    // gif decoder returns 4-D tensor, remove the first dim
    image_reader =
        Squeeze(root.WithOpName("squeeze_first_dim"),
                DecodeGif(root.WithOpName("gif_reader"), file_reader));
  } else if (tensorflow::StringPiece(file_name).ends_with(".bmp")) {
    image_reader = DecodeBmp(root.WithOpName("bmp_reader"), file_reader);
  } else {
    // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
    image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                              DecodeJpeg::Channels(wanted_channels));
  }
  // Now cast the image data to float so we can do normal math on it.
  auto float_caster =
      Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);
  // The convention for image ops in TensorFlow is that all images are expected
  // to be in batches, so that they're four-dimensional arrays with indices of
  // [batch, height, width, channel]. Because we only have a single image, we
  // have to add a batch dimension of 1 to the start with ExpandDims().
  auto dims_expander = ExpandDims(root.WithOpName("expand"), float_caster, 0);
  // Bilinearly resize the image to fit the required dimensions.
 // auto resized = ResizeBilinear(
      //root, dims_expander,
      //Const(root.WithOpName("size"), {input_height, input_width}));
  // Subtract the mean and divide by the scale.
  //Div(root.WithOpName(output_name), Sub(root, resized, {input_mean}),
      //{input_std});
  float input_max = 255;
  Div(root.WithOpName("div"),dims_expander,input_max);
  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({inputs}, {"div"}, {}, out_tensors));
  return Status::OK();
}
int main(int argc, char** argv )
{



  Session* session;
  Status status = NewSession(SessionOptions(), &session);//创建新会话Session


  string model_path="model.pb";
  GraphDef graphdef; //Graph Definition for current model


  Status status_load = ReadBinaryProto(Env::Default(), model_path, &graphdef); //从pb文件中读取图模型;
  if (!status_load.ok()) {
      std::cout << "ERROR: Loading model failed..." << model_path << std::endl;
      std::cout << status_load.ToString() << "\n";
      return -1;
  }
  Status status_create = session->Create(graphdef); //将模型导入会话Session中;
  if (!status_create.ok()) {
      std::cout << "ERROR: Creating graph in session failed..." << status_create.ToString() << std::endl;
      return -1;
  }
  cout << "Session successfully created."<< endl;
  string image_path= argv[1];
  int input_height =28;
  int input_width=28;
  int input_mean=0;
  int input_std=1;
  std::vector<Tensor> resized_tensors;
  Status read_tensor_status =
      ReadTensorFromImageFile(image_path, input_height, input_width, input_mean,
                              input_std, &resized_tensors);
  if (!read_tensor_status.ok()) {
    LOG(ERROR) << read_tensor_status;
    cout<<"resing error"<<endl;
    return -1;
  }

  const Tensor& resized_tensor = resized_tensors[0];
  std::cout << resized_tensor.DebugString()<<endl;


  vector<tensorflow::Tensor> outputs;
  string output_node = "softmax";
  Status status_run = session->Run({{"inputs", resized_tensor}}, {output_node}, {}, &outputs);

  if (!status_run.ok()) {
      std::cout << "ERROR: RUN failed..."  << std::endl;
      std::cout << status_run.ToString() << "\n";
      return -1;
  }
  //Fetch output value
  std::cout << "Output tensor size:" << outputs.size() << std::endl;
  for (std::size_t i = 0; i < outputs.size(); i++) {
      std::cout << outputs[i].DebugString()<<endl;
  }

  Tensor t = outputs[0];                   // Fetch the first tensor
  int ndim2 = t.shape().dims();             // Get the dimension of the tensor
  auto tmap = t.tensor<float, 2>();        // Tensor Shape: [batch_size, target_class_num]
  int output_dim = t.shape().dim_size(1);  // Get the target_class_num from 1st dimension
  std::vector<double> tout;

  // Argmax: Get Final Prediction Label and Probability
  int output_class_id = -1;
  double output_prob = 0.0;
  for (int j = 0; j < output_dim; j++)
  {
        std::cout << "Class " << j << " prob:" << tmap(0, j) << "," << std::endl;
        if (tmap(0, j) >= output_prob) {
              output_class_id = j;
              output_prob = tmap(0, j);
           }
  }

      // Log
  std::cout << "Final class id: " << output_class_id << std::endl;
  std::cout << "Final class prob: " << output_prob << std::endl;

       


     //auto f=t.shaped<float,2>({2,5});
     //cout << f <<endl;
     //Eigen::array<int, 1> reduction_dims{0};

     //cout<< f.maximum(reduction_dims)<<endl;
     //Eigen::array<int, 2> offsets = {0, 0};
     //Eigen::array<int, 2> extents = {2, 1};
     //auto slice = f.slice(offsets, extents);
     //float *index = slice.data();
     //cout << "slice" << endl << slice << endl;
     //cout << slice.argmax()<<endl;
  return 0;
}
# tensorflow-c-mnist
