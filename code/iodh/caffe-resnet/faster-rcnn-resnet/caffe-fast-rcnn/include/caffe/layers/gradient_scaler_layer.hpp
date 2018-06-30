#ifndef CAFFE_MVN_LAYER_HPP_
#define CAFFE_MVN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class GradientScalerLayer: public Layer<Dtype>{

    public:
  explicit GradientScalerLayer(const LayerParameter& param): Layer<Dtype>(param) {}

virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) ;
  float lower_bound_, upper_bound_, alpha_, max_iter_, height_;
  float& coeff_;



};

}  // namespace caffe

#endif  // CAFFE_MVN_LAYER_HPP_
