#include <omp.h>
#include "SemanticSegmentationForests.h"
#include "ConfigReader.h"
#include "utils.h"

void process_tree( vector<uint32_t>::const_iterator p,
		    cv::Rect box, vision::Sample<float> s, vision::ConfigReader *cr,
		    int lPXOff, int lPYOff, vector<cv::Mat>* result)
{
  cv::Point pt = cv::Point();
  int y, x;
  #pragma omp parallel for private(x) private(y) //private(s) private(box) private(lPXOff) private(lPYOff)
  for (y=(int)s.y-lPYOff; y <= (int)s.y+(int)lPYOff; ++y)
	{
	  for (x=(int)s.x-(int)lPXOff; x <= (int)s.x+(int)lPXOff; ++x,++p)
	  {
		  if (*p<0 || *p >= (size_t)cr->numLabels)
		  {
			  std::cerr << "Invalid label in prediction: " << (int) *p << "\n";
			  exit(1);
		  }

		  pt.x = x; pt.y = y;
		  if (box.contains(pt))
		  {
		  (*result)[*p].at<float>(pt) += 1;

		  }
	  }
	}
}
