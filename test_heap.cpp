#include<iostream>
#include "StrucClassSSF.h"

using namespace vision;
int main()
{
	// TODO
		// StructClassSSF.h
			// vector<uint32_t>::const_iterator /* & */ predictPtr(Sample<FeatureType> &sample) const
			// void updateError(IErrorData &newError, const IErrorData &errorData,
			//     const TNode<SplitData<FeatureType>, Prediction> *node, Prediction &newLeft,
			//     Prediction &newRight) const
			// virtual void write(const TNode<SplitData<FeatureType>, Prediction> *node, ostream &out) const
			// virtual void read(TNode<SplitData<FeatureType>, Prediction> *node, istream &in) const
		// RandomHeap.h
			// void split(uint32_t start, uint32_t middle)
			// Prediction predict(Sample &sample) const
			// bool tryImprovingSplit(ErrorData &errorData, TNode<SplitData, Prediction> *node)
			// virtual void write(const TNode<SplitData, Prediction> *node, ostream &out) const
			// virtual void read(TNode<SplitData, Prediction> *node, istream &in) const

    StrucClassSSF<float> *forest = new StrucClassSSF<float>[10];
    //RandomTree* tree = new RandomTree();
    
}
