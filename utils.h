
#ifndef OPENMP_UTILS_H_
#define OPENMP_UTILS_H_

void process_tree(vector<uint32_t>::const_iterator p,
		    cv::Rect box, vision::Sample<float> s, vision::ConfigReader *cr,
		    int lPXOff, int lPYOff, vector<cv::Mat>* result);

#endif
