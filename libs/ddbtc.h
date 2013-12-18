//////////////////////////////////////////////////////////////////////////
//
// SOURCE CODE: https://sites.google.com/site/yunfuliu/2010/source-code---improved-block-truncation-coding-using-optimized-dot-diffusion
//
// CURRENT VERSION: v1.1
//
// CORRESPONDING ARTICLE:
//	Jing-Ming Guo and Yun-Fu Liu, "Improved block truncation coding using optimized dot diffusion," IEEE Trans. Image Processing, to appear.
//
// CONTACT INFO:
//	Yun-Fu Liu (yunfuliu@gmail.com)
//
//////////////////////////////////////////////////////////////////////////

#include <opencv2/core/core.hpp>

namespace ddbtc{

	bool compress(cv::Mat &src,cv::Mat &dst,short BlockSize);

}