// Fillers are random number generators that fills a blob using the specified
// algorithm. The expectation is that they are only going to be used during
// initialization time and will not involve any GPUs.

#ifndef CAFFE_FILLER_HPP
#define CAFFE_FILLER_HPP

#include <string>

#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

/// @brief Fills a Blob with constant or randomly-generated data.
template <typename Dtype>
class Filler {
 public:
  explicit Filler(const FillerParameter& param) : filler_param_(param) {}
  virtual ~Filler() {}
  virtual void Fill(Blob<Dtype>* blob) = 0;
 protected:
  FillerParameter filler_param_;
};  // class Filler


/// @brief Fills a Blob with constant values @f$ x = 0 @f$.
template <typename Dtype>
class ConstantFiller : public Filler<Dtype> {
 public:
  explicit ConstantFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    const int count = blob->count();
    const Dtype value = this->filler_param_.value();
    CHECK(count);
    for (int i = 0; i < count; ++i) {
      data[i] = value;
    }
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/// @brief Fills a Blob with uniformly distributed values @f$ x\sim U(a, b) @f$.
template <typename Dtype>
class UniformFiller : public Filler<Dtype> {
 public:
  explicit UniformFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK(blob->count());
    caffe_rng_uniform<Dtype>(blob->count(), Dtype(this->filler_param_.min()),
        Dtype(this->filler_param_.max()), blob->mutable_cpu_data());
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/// @brief Fills a Blob with Gaussian-distributed values @f$ x = a @f$.
template <typename Dtype>
class GaussianFiller : public Filler<Dtype> {
 public:
  explicit GaussianFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    CHECK(blob->count());
    caffe_rng_gaussian<Dtype>(blob->count(), Dtype(this->filler_param_.mean()),
        Dtype(this->filler_param_.std()), blob->mutable_cpu_data());
    int sparse = this->filler_param_.sparse();
    CHECK_GE(sparse, -1);
    if (sparse >= 0) {
      // Sparse initialization is implemented for "weight" blobs; i.e. matrices.
      // These have num == channels == 1; width is number of inputs; height is
      // number of outputs.  The 'sparse' variable specifies the mean number
      // of non-zero input weights for a given output.
      CHECK_GE(blob->num_axes(), 1);
      const int num_outputs = blob->shape(0);
      Dtype non_zero_probability = Dtype(sparse) / Dtype(num_outputs);
      rand_vec_.reset(new SyncedMemory(blob->count() * sizeof(int)));
      int* mask = reinterpret_cast<int*>(rand_vec_->mutable_cpu_data());
      caffe_rng_bernoulli(blob->count(), non_zero_probability, mask);
      for (int i = 0; i < blob->count(); ++i) {
        data[i] *= mask[i];
      }
    }
  }

 protected:
  shared_ptr<SyncedMemory> rand_vec_;
};

/** @brief Fills a Blob with values @f$ x \in [0, 1] @f$
 *         such that @f$ \forall i \sum_j x_{ij} = 1 @f$.
 */
template <typename Dtype>
class PositiveUnitballFiller : public Filler<Dtype> {
 public:
  explicit PositiveUnitballFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    DCHECK(blob->count());
    caffe_rng_uniform<Dtype>(blob->count(), 0, 1, blob->mutable_cpu_data());
    // We expect the filler to not be called very frequently, so we will
    // just use a simple implementation
    int dim = blob->count() / blob->num();
    CHECK(dim);
    for (int i = 0; i < blob->num(); ++i) {
      Dtype sum = 0;
      for (int j = 0; j < dim; ++j) {
        sum += data[i * dim + j];
      }
      for (int j = 0; j < dim; ++j) {
        data[i * dim + j] /= sum;
      }
    }
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/**
 * @brief Fills a Blob with values @f$ x \sim U(-a, +a) @f$ where @f$ a @f$ is
 *        set inversely proportional to number of incoming nodes, outgoing
 *        nodes, or their average.
 *
 * A Filler based on the paper [Bengio and Glorot 2010]: Understanding
 * the difficulty of training deep feedforward neuralnetworks.
 *
 * It fills the incoming matrix by randomly sampling uniform data from [-scale,
 * scale] where scale = sqrt(3 / n) where n is the fan_in, fan_out, or their
 * average, depending on the variance_norm option. You should make sure the
 * input blob has shape (num, a, b, c) where a * b * c = fan_in and num * b * c
 * = fan_out. Note that this is currently not the case for inner product layers.
 *
 * TODO(dox): make notation in above comment consistent with rest & use LaTeX.
 */
template <typename Dtype>
class XavierFiller : public Filler<Dtype> {
 public:
  explicit XavierFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK(blob->count());
    int fan_in = blob->count() / blob->num();
    int fan_out = blob->count() / blob->channels();
    Dtype n = fan_in;  // default to fan_in
    if (this->filler_param_.variance_norm() ==
        FillerParameter_VarianceNorm_AVERAGE) {
      n = (fan_in + fan_out) / Dtype(2);
    } else if (this->filler_param_.variance_norm() ==
        FillerParameter_VarianceNorm_FAN_OUT) {
      n = fan_out;
    }
    Dtype scale = sqrt(Dtype(3) / n);
    caffe_rng_uniform<Dtype>(blob->count(), -scale, scale,
        blob->mutable_cpu_data());
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/**
 * @brief Fills a Blob with values @f$ x \sim N(0, \sigma^2) @f$ where
 *        @f$ \sigma^2 @f$ is set inversely proportional to number of incoming
 *        nodes, outgoing nodes, or their average.
 *
 * A Filler based on the paper [He, Zhang, Ren and Sun 2015]: Specifically
 * accounts for ReLU nonlinearities.
 *
 * Aside: for another perspective on the scaling factor, see the derivation of
 * [Saxe, McClelland, and Ganguli 2013 (v3)].
 *
 * It fills the incoming matrix by randomly sampling Gaussian data with std =
 * sqrt(2 / n) where n is the fan_in, fan_out, or their average, depending on
 * the variance_norm option. You should make sure the input blob has shape (num,
 * a, b, c) where a * b * c = fan_in and num * b * c = fan_out. Note that this
 * is currently not the case for inner product layers.
 */
template <typename Dtype>
class MSRAFiller : public Filler<Dtype> {
 public:
  explicit MSRAFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK(blob->count());
    int fan_in = blob->count() / blob->num();
    int fan_out = blob->count() / blob->channels();
    Dtype n = fan_in;  // default to fan_in
    if (this->filler_param_.variance_norm() ==
        FillerParameter_VarianceNorm_AVERAGE) {
      n = (fan_in + fan_out) / Dtype(2);
    } else if (this->filler_param_.variance_norm() ==
        FillerParameter_VarianceNorm_FAN_OUT) {
      n = fan_out;
    }
    Dtype std = sqrt(Dtype(2) / n);
    caffe_rng_gaussian<Dtype>(blob->count(), Dtype(0), std,
        blob->mutable_cpu_data());
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/*!
@brief Fills a Blob with coefficients for bilinear interpolation.

A common use case is with the DeconvolutionLayer acting as upsampling.
You can upsample a feature map with shape of (B, C, H, W) by any integer factor
using the following proto.
\code
layer {
  name: "upsample", type: "Deconvolution"
  bottom: "{{bottom_name}}" top: "{{top_name}}"
  convolution_param {
    kernel_size: {{2 * factor - factor % 2}} stride: {{factor}}
    num_output: {{C}} group: {{C}}
    pad: {{ceil((factor - 1) / 2.)}}
    weight_filler: { type: "bilinear" } bias_term: false
  }
  param { lr_mult: 0 decay_mult: 0 }
}
\endcode
Please use this by replacing `{{}}` with your values. By specifying
`num_output: {{C}} group: {{C}}`, it behaves as
channel-wise convolution. The filter shape of this deconvolution layer will be
(C, 1, K, K) where K is `kernel_size`, and this filler will set a (K, K)
interpolation kernel for every channel of the filter identically. The resulting
shape of the top feature map will be (B, C, factor * H, factor * W).
Note that the learning rate and the
weight decay are set to 0 in order to keep coefficient values of bilinear
interpolation unchanged during training. If you apply this to an image, this
operation is equivalent to the following call in Python with Scikit.Image.
\code{.py}
out = skimage.transform.rescale(img, factor, mode='constant', cval=0)
\endcode
 */
template <typename Dtype>
class BilinearFiller : public Filler<Dtype> {
 public:
  explicit BilinearFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK_EQ(blob->num_axes(), 4) << "Blob must be 4 dim.";
    CHECK_EQ(blob->width(), blob->height()) << "Filter must be square";
    Dtype* data = blob->mutable_cpu_data();
    int f = ceil(blob->width() / 2.);
    float c = (2 * f - 1 - f % 2) / (2. * f);
    for (int i = 0; i < blob->count(); ++i) {
      float x = i % blob->width();
      float y = (i / blob->width()) % blob->height();
      data[i] = (1 - fabs(x / f - c)) * (1 - fabs(y / f - c));
    }
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

template <typename Dtype>
class KbFiller : public Filler<Dtype> {
 public:
  explicit KbFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
	Dtype* weights = blob->mutable_cpu_data();
	weights[0] = -1.0/4.0;
	weights[1] = 2.0/4.0;
	weights[2] = -1.0/4.0;
	weights[3] = 2.0/4.0;
	weights[4] = -4.0/4.0;
	weights[5] = 2.0/4.0;
	weights[6] = -1.0/4.0;
	weights[7] = 2.0/4.0;
	weights[8] = -1.0/4.0;
  }
};

template <typename Dtype>
class QianFiller : public Filler<Dtype> {
 public:
  explicit QianFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
	Dtype* weights = blob->mutable_cpu_data();
	weights[0] = -1.0/12.0;
	weights[1] = 2.0/12.0;
	weights[2] = -2.0/12.0;
	weights[3] = 2.0/12.0;
	weights[4] = -1.0/12.0;
	weights[5] = 2.0/12.0;
	weights[6] = -6.0/12.0;
	weights[7] = 8.0/12.0;
	weights[8] = -6.0/12.0;
	weights[9] = 2.0/12.0;
	weights[10] = -2.0/12.0;
	weights[11] = 8.0/12.0;
	weights[12] = -12.0/12.0;
	weights[13] = 8.0/12.0;
	weights[14] = -2.0/12.0;
	weights[15] = 2.0/12.0;
	weights[16] = -6.0/12.0;
	weights[17] = 8.0/12.0;
	weights[18] = -6.0/12.0;
	weights[19] = 2.0/12.0;
	weights[20] = -1.0/12.0;
	weights[21] = 2.0/12.0;
	weights[22] = -2.0/12.0;
	weights[23] = 2.0/12.0;
	weights[24] = -1.0/12.0;
  }
};

  /*DctrFiller filler is copy from DCTR,and used to generate 9 DCT bases*/
template <typename Dtype>
class DctrFiller : public Filler<Dtype> {
 public:
  explicit DctrFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    float PI = 3.141592654f;
    Dtype* weights = blob->mutable_cpu_data();
    int mode_r;
    int mode_c;
    int m;
    int n;

    for (int i=0;i<blob->num();i++){
      mode_r=i/3;
      mode_c=i%3;
      for(int j=0;j<9;j++){
	m=j/3;
	n=j%3;
	float wr = sqrt(2.0f); if (mode_r==0) wr=1;
	float wc = sqrt(2.0f); if (mode_c==0) wc=1;
	float val = ((wr*wc)/3) * cos(PI*mode_r*(2*m+1) / 6) * cos(PI*mode_c*(2*n+1) / 6);
	weights[i*9+j] = val;
      }
    }
  }
};

  /*SrmFiller filler is copy from SRM,and used to generate 30 srm kernels, each kernels size is 5x5*/
template <typename Dtype>
class SrmFiller : public Filler<Dtype> {
 public:
  explicit SrmFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
	Dtype* weights = blob->mutable_cpu_data();
	        //kernel 1 start!
        weights[0] =   0.00000000;
        weights[1] =   0.00000000;
        weights[2] =   0.00000000;
        weights[3] =   0.00000000;
        weights[4] =   0.00000000;
        weights[5] =   0.00000000;
        weights[6] =   0.00000000;
        weights[7] =   0.00000000;
        weights[8] =   0.00000000;
        weights[9] =   0.00000000;
        weights[10] =   0.00000000;
        weights[11] =   0.00000000;
        weights[12] =  -1.00000000;
        weights[13] =   1.00000000;
        weights[14] =   0.00000000;
        weights[15] =   0.00000000;
        weights[16] =   0.00000000;
        weights[17] =   0.00000000;
        weights[18] =   0.00000000;
        weights[19] =   0.00000000;
        weights[20] =   0.00000000;
        weights[21] =   0.00000000;
        weights[22] =   0.00000000;
        weights[23] =   0.00000000;
        weights[24] =   0.00000000;
        //kernel 1 end!

        //kernel 2 start!
        weights[25] =   0.00000000;
        weights[26] =   0.00000000;
        weights[27] =   0.00000000;
        weights[28] =   0.00000000;
        weights[29] =   0.00000000;
        weights[30] =   0.00000000;
        weights[31] =   0.00000000;
        weights[32] =   0.00000000;
        weights[33] =   0.00000000;
        weights[34] =   0.00000000;
        weights[35] =   0.00000000;
        weights[36] =   1.00000000;
        weights[37] =  -1.00000000;
        weights[38] =   0.00000000;
        weights[39] =   0.00000000;
        weights[40] =   0.00000000;
        weights[41] =   0.00000000;
        weights[42] =   0.00000000;
        weights[43] =   0.00000000;
        weights[44] =   0.00000000;
        weights[45] =   0.00000000;
        weights[46] =   0.00000000;
        weights[47] =   0.00000000;
        weights[48] =   0.00000000;
        weights[49] =   0.00000000;
        //kernel 2 end!

        //kernel 3 start!
        weights[50] =   0.00000000;
        weights[51] =   0.00000000;
        weights[52] =   0.00000000;
        weights[53] =   0.00000000;
        weights[54] =   0.00000000;
        weights[55] =   0.00000000;
        weights[56] =   0.00000000;
        weights[57] =   1.00000000;
        weights[58] =   0.00000000;
        weights[59] =   0.00000000;
        weights[60] =   0.00000000;
        weights[61] =   0.00000000;
        weights[62] =  -1.00000000;
        weights[63] =   0.00000000;
        weights[64] =   0.00000000;
        weights[65] =   0.00000000;
        weights[66] =   0.00000000;
        weights[67] =   0.00000000;
        weights[68] =   0.00000000;
        weights[69] =   0.00000000;
        weights[70] =   0.00000000;
        weights[71] =   0.00000000;
        weights[72] =   0.00000000;
        weights[73] =   0.00000000;
        weights[74] =   0.00000000;
        //kernel 3 end!

        //kernel 4 start!
        weights[75] =   0.00000000;
        weights[76] =   0.00000000;
        weights[77] =   0.00000000;
        weights[78] =   0.00000000;
        weights[79] =   0.00000000;
        weights[80] =   0.00000000;
        weights[81] =   0.00000000;
        weights[82] =   0.00000000;
        weights[83] =   0.00000000;
        weights[84] =   0.00000000;
        weights[85] =   0.00000000;
        weights[86] =   0.00000000;
        weights[87] =  -1.00000000;
        weights[88] =   0.00000000;
        weights[89] =   0.00000000;
        weights[90] =   0.00000000;
        weights[91] =   0.00000000;
        weights[92] =   1.00000000;
        weights[93] =   0.00000000;
        weights[94] =   0.00000000;
        weights[95] =   0.00000000;
        weights[96] =   0.00000000;
        weights[97] =   0.00000000;
        weights[98] =   0.00000000;
        weights[99] =   0.00000000;
        //kernel 4 end!

        //kernel 5 start!
        weights[100] =   0.00000000;
        weights[101] =   0.00000000;
        weights[102] =   0.00000000;
        weights[103] =   0.00000000;
        weights[104] =   0.00000000;
        weights[105] =   0.00000000;
        weights[106] =   0.00000000;
        weights[107] =   0.00000000;
        weights[108] =   1.00000000;
        weights[109] =   0.00000000;
        weights[110] =   0.00000000;
        weights[111] =   0.00000000;
        weights[112] =  -1.00000000;
        weights[113] =   0.00000000;
        weights[114] =   0.00000000;
        weights[115] =   0.00000000;
        weights[116] =   0.00000000;
        weights[117] =   0.00000000;
        weights[118] =   0.00000000;
        weights[119] =   0.00000000;
        weights[120] =   0.00000000;
        weights[121] =   0.00000000;
        weights[122] =   0.00000000;
        weights[123] =   0.00000000;
        weights[124] =   0.00000000;
        //kernel 5 end!

        //kernel 6 start!
        weights[125] =   0.00000000;
        weights[126] =   0.00000000;
        weights[127] =   0.00000000;
        weights[128] =   0.00000000;
        weights[129] =   0.00000000;
        weights[130] =   0.00000000;
        weights[131] =   0.00000000;
        weights[132] =   0.00000000;
        weights[133] =   0.00000000;
        weights[134] =   0.00000000;
        weights[135] =   0.00000000;
        weights[136] =   0.00000000;
        weights[137] =  -1.00000000;
        weights[138] =   0.00000000;
        weights[139] =   0.00000000;
        weights[140] =   0.00000000;
        weights[141] =   0.00000000;
        weights[142] =   0.00000000;
        weights[143] =   1.00000000;
        weights[144] =   0.00000000;
        weights[145] =   0.00000000;
        weights[146] =   0.00000000;
        weights[147] =   0.00000000;
        weights[148] =   0.00000000;
        weights[149] =   0.00000000;
        //kernel 6 end!

        //kernel 7 start!
        weights[150] =   0.00000000;
        weights[151] =   0.00000000;
        weights[152] =   0.00000000;
        weights[153] =   0.00000000;
        weights[154] =   0.00000000;
        weights[155] =   0.00000000;
        weights[156] =   1.00000000;
        weights[157] =   0.00000000;
        weights[158] =   0.00000000;
        weights[159] =   0.00000000;
        weights[160] =   0.00000000;
        weights[161] =   0.00000000;
        weights[162] =  -1.00000000;
        weights[163] =   0.00000000;
        weights[164] =   0.00000000;
        weights[165] =   0.00000000;
        weights[166] =   0.00000000;
        weights[167] =   0.00000000;
        weights[168] =   0.00000000;
        weights[169] =   0.00000000;
        weights[170] =   0.00000000;
        weights[171] =   0.00000000;
        weights[172] =   0.00000000;
        weights[173] =   0.00000000;
        weights[174] =   0.00000000;
        //kernel 7 end!

        //kernel 8 start!
        weights[175] =   0.00000000;
        weights[176] =   0.00000000;
        weights[177] =   0.00000000;
        weights[178] =   0.00000000;
        weights[179] =   0.00000000;
        weights[180] =   0.00000000;
        weights[181] =   0.00000000;
        weights[182] =   0.00000000;
        weights[183] =   0.00000000;
        weights[184] =   0.00000000;
        weights[185] =   0.00000000;
        weights[186] =   0.00000000;
        weights[187] =  -1.00000000;
        weights[188] =   0.00000000;
        weights[189] =   0.00000000;
        weights[190] =   0.00000000;
        weights[191] =   1.00000000;
        weights[192] =   0.00000000;
        weights[193] =   0.00000000;
        weights[194] =   0.00000000;
        weights[195] =   0.00000000;
        weights[196] =   0.00000000;
        weights[197] =   0.00000000;
        weights[198] =   0.00000000;
        weights[199] =   0.00000000;
        //kernel 8 end!

        //kernel 9 start!
        weights[200] =   0.00000000;
        weights[201] =   0.00000000;
        weights[202] =   0.00000000;
        weights[203] =   0.00000000;
        weights[204] =   0.00000000;
        weights[205] =   0.00000000;
        weights[206] =   0.00000000;
        weights[207] =   0.00000000;
        weights[208] =   0.00000000;
        weights[209] =   0.00000000;
        weights[210] =   0.00000000;
        weights[211] =   0.50000000;
        weights[212] =  -1.00000000;
        weights[213] =   0.50000000;
        weights[214] =   0.00000000;
        weights[215] =   0.00000000;
        weights[216] =   0.00000000;
        weights[217] =   0.00000000;
        weights[218] =   0.00000000;
        weights[219] =   0.00000000;
        weights[220] =   0.00000000;
        weights[221] =   0.00000000;
        weights[222] =   0.00000000;
        weights[223] =   0.00000000;
        weights[224] =   0.00000000;
        //kernel 9 end!

        //kernel 10 start!
        weights[225] =   0.00000000;
        weights[226] =   0.00000000;
        weights[227] =   0.00000000;
        weights[228] =   0.00000000;
        weights[229] =   0.00000000;
        weights[230] =   0.00000000;
        weights[231] =   0.00000000;
        weights[232] =   0.50000000;
        weights[233] =   0.00000000;
        weights[234] =   0.00000000;
        weights[235] =   0.00000000;
        weights[236] =   0.00000000;
        weights[237] =  -1.00000000;
        weights[238] =   0.00000000;
        weights[239] =   0.00000000;
        weights[240] =   0.00000000;
        weights[241] =   0.00000000;
        weights[242] =   0.50000000;
        weights[243] =   0.00000000;
        weights[244] =   0.00000000;
        weights[245] =   0.00000000;
        weights[246] =   0.00000000;
        weights[247] =   0.00000000;
        weights[248] =   0.00000000;
        weights[249] =   0.00000000;
        //kernel 10 end!

        //kernel 11 start!
        weights[250] =   0.00000000;
        weights[251] =   0.00000000;
        weights[252] =   0.00000000;
        weights[253] =   0.00000000;
        weights[254] =   0.00000000;
        weights[255] =   0.00000000;
        weights[256] =   0.50000000;
        weights[257] =   0.00000000;
        weights[258] =   0.00000000;
        weights[259] =   0.00000000;
        weights[260] =   0.00000000;
        weights[261] =   0.00000000;
        weights[262] =  -1.00000000;
        weights[263] =   0.00000000;
        weights[264] =   0.00000000;
        weights[265] =   0.00000000;
        weights[266] =   0.00000000;
        weights[267] =   0.00000000;
        weights[268] =   0.50000000;
        weights[269] =   0.00000000;
        weights[270] =   0.00000000;
        weights[271] =   0.00000000;
        weights[272] =   0.00000000;
        weights[273] =   0.00000000;
        weights[274] =   0.00000000;
        //kernel 11 end!

        //kernel 12 start!
        weights[275] =   0.00000000;
        weights[276] =   0.00000000;
        weights[277] =   0.00000000;
        weights[278] =   0.00000000;
        weights[279] =   0.00000000;
        weights[280] =   0.00000000;
        weights[281] =   0.00000000;
        weights[282] =   0.00000000;
        weights[283] =   0.50000000;
        weights[284] =   0.00000000;
        weights[285] =   0.00000000;
        weights[286] =   0.00000000;
        weights[287] =  -1.00000000;
        weights[288] =   0.00000000;
        weights[289] =   0.00000000;
        weights[290] =   0.00000000;
        weights[291] =   0.50000000;
        weights[292] =   0.00000000;
        weights[293] =   0.00000000;
        weights[294] =   0.00000000;
        weights[295] =   0.00000000;
        weights[296] =   0.00000000;
        weights[297] =   0.00000000;
        weights[298] =   0.00000000;
        weights[299] =   0.00000000;
        //kernel 12 end!

        //kernel 13 start!
        weights[300] =   0.00000000;
        weights[301] =   0.00000000;
        weights[302] =   0.00000000;
        weights[303] =   0.00000000;
        weights[304] =   0.00000000;
        weights[305] =   0.00000000;
        weights[306] =   0.00000000;
        weights[307] =   0.00000000;
        weights[308] =   0.00000000;
        weights[309] =   0.00000000;
        weights[310] =   0.00000000;
        weights[311] =   0.33333333;
        weights[312] =  -1.00000000;
        weights[313] =   1.00000000;
        weights[314] =  -0.33333333;
        weights[315] =   0.00000000;
        weights[316] =   0.00000000;
        weights[317] =   0.00000000;
        weights[318] =   0.00000000;
        weights[319] =   0.00000000;
        weights[320] =   0.00000000;
        weights[321] =   0.00000000;
        weights[322] =   0.00000000;
        weights[323] =   0.00000000;
        weights[324] =   0.00000000;
        //kernel 13 end!

        //kernel 14 start!
        weights[325] =   0.00000000;
        weights[326] =   0.00000000;
        weights[327] =   0.00000000;
        weights[328] =   0.00000000;
        weights[329] =   0.00000000;
        weights[330] =   0.00000000;
        weights[331] =   0.00000000;
        weights[332] =   0.00000000;
        weights[333] =   0.00000000;
        weights[334] =   0.00000000;
        weights[335] =  -0.33333333;
        weights[336] =   1.00000000;
        weights[337] =  -1.00000000;
        weights[338] =   0.33333333;
        weights[339] =   0.00000000;
        weights[340] =   0.00000000;
        weights[341] =   0.00000000;
        weights[342] =   0.00000000;
        weights[343] =   0.00000000;
        weights[344] =   0.00000000;
        weights[345] =   0.00000000;
        weights[346] =   0.00000000;
        weights[347] =   0.00000000;
        weights[348] =   0.00000000;
        weights[349] =   0.00000000;
        //kernel 14 end!

        //kernel 15 start!
        weights[350] =   0.00000000;
        weights[351] =   0.00000000;
        weights[352] =  -0.33333333;
        weights[353] =   0.00000000;
        weights[354] =   0.00000000;
        weights[355] =   0.00000000;
        weights[356] =   0.00000000;
        weights[357] =   1.00000000;
        weights[358] =   0.00000000;
        weights[359] =   0.00000000;
        weights[360] =   0.00000000;
        weights[361] =   0.00000000;
        weights[362] =  -1.00000000;
        weights[363] =   0.00000000;
        weights[364] =   0.00000000;
        weights[365] =   0.00000000;
        weights[366] =   0.00000000;
        weights[367] =   0.33333333;
        weights[368] =   0.00000000;
        weights[369] =   0.00000000;
        weights[370] =   0.00000000;
        weights[371] =   0.00000000;
        weights[372] =   0.00000000;
        weights[373] =   0.00000000;
        weights[374] =   0.00000000;
        //kernel 15 end!

        //kernel 16 start!
        weights[375] =   0.00000000;
        weights[376] =   0.00000000;
        weights[377] =   0.00000000;
        weights[378] =   0.00000000;
        weights[379] =   0.00000000;
        weights[380] =   0.00000000;
        weights[381] =   0.00000000;
        weights[382] =   0.33333333;
        weights[383] =   0.00000000;
        weights[384] =   0.00000000;
        weights[385] =   0.00000000;
        weights[386] =   0.00000000;
        weights[387] =  -1.00000000;
        weights[388] =   0.00000000;
        weights[389] =   0.00000000;
        weights[390] =   0.00000000;
        weights[391] =   0.00000000;
        weights[392] =   1.00000000;
        weights[393] =   0.00000000;
        weights[394] =   0.00000000;
        weights[395] =   0.00000000;
        weights[396] =   0.00000000;
        weights[397] =  -0.33333333;
        weights[398] =   0.00000000;
        weights[399] =   0.00000000;
        //kernel 16 end!

        //kernel 17 start!
        weights[400] =   0.00000000;
        weights[401] =   0.00000000;
        weights[402] =   0.00000000;
        weights[403] =   0.00000000;
        weights[404] =  -0.33333333;
        weights[405] =   0.00000000;
        weights[406] =   0.00000000;
        weights[407] =   0.00000000;
        weights[408] =   1.00000000;
        weights[409] =   0.00000000;
        weights[410] =   0.00000000;
        weights[411] =   0.00000000;
        weights[412] =  -1.00000000;
        weights[413] =   0.00000000;
        weights[414] =   0.00000000;
        weights[415] =   0.00000000;
        weights[416] =   0.33333333;
        weights[417] =   0.00000000;
        weights[418] =   0.00000000;
        weights[419] =   0.00000000;
        weights[420] =   0.00000000;
        weights[421] =   0.00000000;
        weights[422] =   0.00000000;
        weights[423] =   0.00000000;
        weights[424] =   0.00000000;
        //kernel 17 end!

        //kernel 18 start!
        weights[425] =   0.00000000;
        weights[426] =   0.00000000;
        weights[427] =   0.00000000;
        weights[428] =   0.00000000;
        weights[429] =   0.00000000;
        weights[430] =   0.00000000;
        weights[431] =   0.33333333;
        weights[432] =   0.00000000;
        weights[433] =   0.00000000;
        weights[434] =   0.00000000;
        weights[435] =   0.00000000;
        weights[436] =   0.00000000;
        weights[437] =  -1.00000000;
        weights[438] =   0.00000000;
        weights[439] =   0.00000000;
        weights[440] =   0.00000000;
        weights[441] =   0.00000000;
        weights[442] =   0.00000000;
        weights[443] =   1.00000000;
        weights[444] =   0.00000000;
        weights[445] =   0.00000000;
        weights[446] =   0.00000000;
        weights[447] =   0.00000000;
        weights[448] =   0.00000000;
        weights[449] =  -0.33333333;
        //kernel 18 end!

        //kernel 19 start!
        weights[450] =  -0.33333333;
        weights[451] =   0.00000000;
        weights[452] =   0.00000000;
        weights[453] =   0.00000000;
        weights[454] =   0.00000000;
        weights[455] =   0.00000000;
        weights[456] =   1.00000000;
        weights[457] =   0.00000000;
        weights[458] =   0.00000000;
        weights[459] =   0.00000000;
        weights[460] =   0.00000000;
        weights[461] =   0.00000000;
        weights[462] =  -1.00000000;
        weights[463] =   0.00000000;
        weights[464] =   0.00000000;
        weights[465] =   0.00000000;
        weights[466] =   0.00000000;
        weights[467] =   0.00000000;
        weights[468] =   0.33333333;
        weights[469] =   0.00000000;
        weights[470] =   0.00000000;
        weights[471] =   0.00000000;
        weights[472] =   0.00000000;
        weights[473] =   0.00000000;
        weights[474] =   0.00000000;
        //kernel 19 end!

        //kernel 20 start!
        weights[475] =   0.00000000;
        weights[476] =   0.00000000;
        weights[477] =   0.00000000;
        weights[478] =   0.00000000;
        weights[479] =   0.00000000;
        weights[480] =   0.00000000;
        weights[481] =   0.00000000;
        weights[482] =   0.00000000;
        weights[483] =   0.33333333;
        weights[484] =   0.00000000;
        weights[485] =   0.00000000;
        weights[486] =   0.00000000;
        weights[487] =  -1.00000000;
        weights[488] =   0.00000000;
        weights[489] =   0.00000000;
        weights[490] =   0.00000000;
        weights[491] =   1.00000000;
        weights[492] =   0.00000000;
        weights[493] =   0.00000000;
        weights[494] =   0.00000000;
        weights[495] =  -0.33333333;
        weights[496] =   0.00000000;
        weights[497] =   0.00000000;
        weights[498] =   0.00000000;
        weights[499] =   0.00000000;
        //kernel 20 end!

        //kernel 21 start!
        weights[500] =   0.00000000;
        weights[501] =   0.00000000;
        weights[502] =   0.00000000;
        weights[503] =   0.00000000;
        weights[504] =   0.00000000;
        weights[505] =   0.00000000;
        weights[506] =   0.00000000;
        weights[507] =   0.50000000;
        weights[508] =  -0.25000000;
        weights[509] =   0.00000000;
        weights[510] =   0.00000000;
        weights[511] =   0.00000000;
        weights[512] =  -1.00000000;
        weights[513] =   0.50000000;
        weights[514] =   0.00000000;
        weights[515] =   0.00000000;
        weights[516] =   0.00000000;
        weights[517] =   0.50000000;
        weights[518] =  -0.25000000;
        weights[519] =   0.00000000;
        weights[520] =   0.00000000;
        weights[521] =   0.00000000;
        weights[522] =   0.00000000;
        weights[523] =   0.00000000;
        weights[524] =   0.00000000;
        //kernel 21 end!

        //kernel 22 start!
        weights[525] =   0.00000000;
        weights[526] =   0.00000000;
        weights[527] =   0.00000000;
        weights[528] =   0.00000000;
        weights[529] =   0.00000000;
        weights[530] =   0.00000000;
        weights[531] =  -0.25000000;
        weights[532] =   0.50000000;
        weights[533] =   0.00000000;
        weights[534] =   0.00000000;
        weights[535] =   0.00000000;
        weights[536] =   0.50000000;
        weights[537] =  -1.00000000;
        weights[538] =   0.00000000;
        weights[539] =   0.00000000;
        weights[540] =   0.00000000;
        weights[541] =  -0.25000000;
        weights[542] =   0.50000000;
        weights[543] =   0.00000000;
        weights[544] =   0.00000000;
        weights[545] =   0.00000000;
        weights[546] =   0.00000000;
        weights[547] =   0.00000000;
        weights[548] =   0.00000000;
        weights[549] =   0.00000000;
        //kernel 22 end!

        //kernel 23 start!
        weights[550] =   0.00000000;
        weights[551] =   0.00000000;
        weights[552] =   0.00000000;
        weights[553] =   0.00000000;
        weights[554] =   0.00000000;
        weights[555] =   0.00000000;
        weights[556] =  -0.25000000;
        weights[557] =   0.50000000;
        weights[558] =  -0.25000000;
        weights[559] =   0.00000000;
        weights[560] =   0.00000000;
        weights[561] =   0.50000000;
        weights[562] =  -1.00000000;
        weights[563] =   0.50000000;
        weights[564] =   0.00000000;
        weights[565] =   0.00000000;
        weights[566] =   0.00000000;
        weights[567] =   0.00000000;
        weights[568] =   0.00000000;
        weights[569] =   0.00000000;
        weights[570] =   0.00000000;
        weights[571] =   0.00000000;
        weights[572] =   0.00000000;
        weights[573] =   0.00000000;
        weights[574] =   0.00000000;
        //kernel 23 end!

        //kernel 24 start!
        weights[575] =   0.00000000;
        weights[576] =   0.00000000;
        weights[577] =   0.00000000;
        weights[578] =   0.00000000;
        weights[579] =   0.00000000;
        weights[580] =   0.00000000;
        weights[581] =   0.00000000;
        weights[582] =   0.00000000;
        weights[583] =   0.00000000;
        weights[584] =   0.00000000;
        weights[585] =   0.00000000;
        weights[586] =   0.50000000;
        weights[587] =  -1.00000000;
        weights[588] =   0.50000000;
        weights[589] =   0.00000000;
        weights[590] =   0.00000000;
        weights[591] =  -0.25000000;
        weights[592] =   0.50000000;
        weights[593] =  -0.25000000;
        weights[594] =   0.00000000;
        weights[595] =   0.00000000;
        weights[596] =   0.00000000;
        weights[597] =   0.00000000;
        weights[598] =   0.00000000;
        weights[599] =   0.00000000;
        //kernel 24 end!

        //kernel 25 start!
        weights[600] =   0.00000000;
        weights[601] =   0.00000000;
        weights[602] =   0.00000000;
        weights[603] =   0.00000000;
        weights[604] =   0.00000000;
        weights[605] =   0.00000000;
        weights[606] =  -0.25000000;
        weights[607] =   0.50000000;
        weights[608] =  -0.25000000;
        weights[609] =   0.00000000;
        weights[610] =   0.00000000;
        weights[611] =   0.50000000;
        weights[612] =  -1.00000000;
        weights[613] =   0.50000000;
        weights[614] =   0.00000000;
        weights[615] =   0.00000000;
        weights[616] =  -0.25000000;
        weights[617] =   0.50000000;
        weights[618] =  -0.25000000;
        weights[619] =   0.00000000;
        weights[620] =   0.00000000;
        weights[621] =   0.00000000;
        weights[622] =   0.00000000;
        weights[623] =   0.00000000;
        weights[624] =   0.00000000;
        //kernel 25 end!

        //kernel 26 start!
        weights[625] =   0.00000000;
        weights[626] =   0.00000000;
        weights[627] =  -0.16666667;
        weights[628] =   0.16666667;
        weights[629] =  -0.08333333;
        weights[630] =   0.00000000;
        weights[631] =   0.00000000;
        weights[632] =   0.66666667;
        weights[633] =  -0.50000000;
        weights[634] =   0.16666667;
        weights[635] =   0.00000000;
        weights[636] =   0.00000000;
        weights[637] =  -1.00000000;
        weights[638] =   0.66666667;
        weights[639] =  -0.16666667;
        weights[640] =   0.00000000;
        weights[641] =   0.00000000;
        weights[642] =   0.66666667;
        weights[643] =  -0.50000000;
        weights[644] =   0.16666667;
        weights[645] =   0.00000000;
        weights[646] =   0.00000000;
        weights[647] =  -0.16666667;
        weights[648] =   0.16666667;
        weights[649] =  -0.08333333;
        //kernel 26 end!

        //kernel 27 start!
        weights[650] =  -0.08333333;
        weights[651] =   0.16666667;
        weights[652] =  -0.16666667;
        weights[653] =   0.00000000;
        weights[654] =   0.00000000;
        weights[655] =   0.16666667;
        weights[656] =  -0.50000000;
        weights[657] =   0.66666667;
        weights[658] =   0.00000000;
        weights[659] =   0.00000000;
        weights[660] =  -0.16666667;
        weights[661] =   0.66666667;
        weights[662] =  -1.00000000;
        weights[663] =   0.00000000;
        weights[664] =   0.00000000;
        weights[665] =   0.16666667;
        weights[666] =  -0.50000000;
        weights[667] =   0.66666667;
        weights[668] =  -0.00000000;
        weights[669] =   0.00000000;
        weights[670] =  -0.08333333;
        weights[671] =   0.16666667;
        weights[672] =  -0.16666667;
        weights[673] =   0.00000000;
        weights[674] =   0.00000000;
        //kernel 27 end!

        //kernel 28 start!
        weights[675] =  -0.08333333;
        weights[676] =   0.16666667;
        weights[677] =  -0.16666667;
        weights[678] =   0.16666667;
        weights[679] =  -0.08333333;
        weights[680] =   0.16666667;
        weights[681] =  -0.50000000;
        weights[682] =   0.66666667;
        weights[683] =  -0.50000000;
        weights[684] =   0.16666667;
        weights[685] =  -0.16666667;
        weights[686] =   0.66666667;
        weights[687] =  -1.00000000;
        weights[688] =   0.66666667;
        weights[689] =  -0.16666667;
        weights[690] =   0.00000000;
        weights[691] =   0.00000000;
        weights[692] =   0.00000000;
        weights[693] =   0.00000000;
        weights[694] =   0.00000000;
        weights[695] =   0.00000000;
        weights[696] =   0.00000000;
        weights[697] =   0.00000000;
        weights[698] =   0.00000000;
        weights[699] =   0.00000000;
        //kernel 28 end!

        //kernel 29 start!
        weights[700] =   0.00000000;
        weights[701] =   0.00000000;
        weights[702] =   0.00000000;
        weights[703] =   0.00000000;
        weights[704] =   0.00000000;
        weights[705] =   0.00000000;
        weights[706] =   0.00000000;
        weights[707] =   0.00000000;
        weights[708] =   0.00000000;
        weights[709] =   0.00000000;
        weights[710] =  -0.16666667;
        weights[711] =   0.66666667;
        weights[712] =  -1.00000000;
        weights[713] =   0.66666667;
        weights[714] =  -0.16666667;
        weights[715] =   0.16666667;
        weights[716] =  -0.50000000;
        weights[717] =   0.66666667;
        weights[718] =  -0.50000000;
        weights[719] =   0.16666667;
        weights[720] =  -0.08333333;
        weights[721] =   0.16666667;
        weights[722] =  -0.16666667;
        weights[723] =   0.16666667;
        weights[724] =  -0.08333333;
        //kernel 29 end!

        //kernel 30 start!
        weights[725] =  -0.08333333;
        weights[726] =   0.16666667;
        weights[727] =  -0.16666667;
        weights[728] =   0.16666667;
        weights[729] =  -0.08333333;
        weights[730] =   0.16666667;
        weights[731] =  -0.50000000;
        weights[732] =   0.66666667;
        weights[733] =  -0.50000000;
        weights[734] =   0.16666667;
        weights[735] =  -0.16666667;
        weights[736] =   0.66666667;
        weights[737] =  -1.00000000;
        weights[738] =   0.66666667;
        weights[739] =  -0.16666667;
        weights[740] =   0.16666667;
        weights[741] =  -0.50000000;
        weights[742] =   0.66666667;
        weights[743] =  -0.50000000;
        weights[744] =   0.16666667;
        weights[745] =  -0.08333333;
        weights[746] =   0.16666667;
        weights[747] =  -0.16666667;
        weights[748] =   0.16666667;
        weights[749] =  -0.08333333;
        //kernel 30 end!
  }
};

/**
 * @brief Get a specific filler from the specification given in FillerParameter.
 *
 * Ideally this would be replaced by a factory pattern, but we will leave it
 * this way for now.
 */
template<typename Dtype>
Filler<Dtype>* GetFiller(const FillerParameter& param) {
	const std::string& type = param.type();
	if (type == "constant") {
		return new ConstantFiller<Dtype>(param);
	} else if (type == "gaussian") {
		return new GaussianFiller<Dtype>(param);
	} else if (type == "positive_unitball") {
		return new PositiveUnitballFiller<Dtype>(param);
	} else if (type == "uniform") {
		return new UniformFiller<Dtype>(param);
	} else if (type == "xavier") {
		return new XavierFiller<Dtype>(param);
	} else if (type == "msra") {
		return new MSRAFiller<Dtype>(param);
	} else if (type == "bilinear") {
		return new BilinearFiller<Dtype>(param);
	} else if (type == "kb") {
		return new KbFiller<Dtype>(param);
	} else if (type == "qian") {
		return new QianFiller<Dtype>(param);
	} else if (type == "srm") {
		return new SrmFiller<Dtype>(param);
	} else if (type == "dctr") {
	        return new DctrFiller<Dtype>(param);
	} else {
		CHECK(false) << "Unknown filler name: " << param.type();
	}
	return (Filler<Dtype>*) (NULL);
}

}  // namespace caffe

#endif  // CAFFE_FILLER_HPP_
