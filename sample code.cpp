#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "libs/ddbtc.h"

void main(){

	cv::Mat	src,dst;

	// read image
	src	=	cv::imread("lake.bmp",CV_LOAD_IMAGE_GRAYSCALE);

	// process (only block sizes 8 and 16 are available for this function)
	if(!ddbtc::compress(src,dst,16)){
		std::cout	<<	"Wrong parameter."	<<	std::endl;
	}else{
		// save image 
		imwrite("result.bmp",dst);

		// display images
		cv::namedWindow("src");
		cv::namedWindow("dst");
		cv::moveWindow("src",0,0);
		cv::moveWindow("dst",src.cols,0);
		cv::imshow("src",src);
		cv::imshow("dst",dst);
	}

	cv::waitKey(0);
}