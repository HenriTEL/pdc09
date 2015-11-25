#include<iostream>
#include<string>
#include "StrucClassSSF.h"

using namespace vision;

// TODO
	// StructClassSSF.h
		// Henri
			// vector<uint32_t>::const_iterator /* & */ predictPtr(Sample<FeatureType> &sample) const
		// Karim
			// virtual void write(const TNode<SplitData<FeatureType>, Prediction> *node, ostream &out) const
			// virtual void read(TNode<SplitData<FeatureType>, Prediction> *node, istream &in) const
	// RandomHeap.h
		// Henri
			// void split(uint32_t start, uint32_t middle)
			// Prediction predict(Sample &sample) const
		// Karim
			// bool tryImprovingSplit(ErrorData &errorData, TNode<SplitData, Prediction> *node)
			// virtual void write(const TNode<SplitData, Prediction> *node, ostream &out) const
			// virtual void read(TNode<SplitData, Prediction> *node, istream &in) const

bool test_creation()
{
	StrucClassSSF<float> *forest = new StrucClassSSF<float>[10];
	return true;
}


int main()
{
	string res = "SUCCESS";
    StrucClassSSF<float> *forest = new StrucClassSSF<float>[10];
    std::cout << "test_creation ";
    if( !test_creation() ) res = "FAIL";
    std::cout << res << std::endl;
    
}
// #ifdef DOGPU
// loadTreeToGPU()
// predictPtr : kernel
