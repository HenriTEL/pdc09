// =========================================================================================
//    Structured Class-Label in Random Forests. This is a re-implementation of
//    the work we presented at ICCV'11 in Barcelona, Spain.
//
//    In case of using this code, please cite the following paper:
//    P. Kontschieder, S. Rota Bulò, H. Bischof and M. Pelillo.
//    Structured Class-Labels in Random Forests for Semantic Image Labelling. In (ICCV), 2011.
//
//    Implementation by Peter Kontschieder and Samuel Rota Bulò
//    October 2013
//
// =========================================================================================

#ifndef RANDOMHEAP_H
#define RANDOMHEAP_H

#include <vector>
#include <cassert>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <stdint.h>

#include "Global.h"

using namespace std;

namespace vision
{

// =====================================================================================
//        Class:  TNode
//  Description:
// =====================================================================================
template<class SplitData, class Prediction>
class TNode
{
public:
  // ====================  LIFECYCLE     =======================================
  TNode(int start, int end) :
      start(start), end(end), depth(0)
  {
    static int iNode = 0;
    idx = iNode++;

    // cout<<endl<<"New node "<<idx<<" "<<hex<<this<<dec<<endl;
  }

  ~TNode()
  {
  }

  // ====================  ACCESSORS     =======================================
  int getStart() const
  {
    return start;
  }
  int getEnd() const
  {
    return end;
  }

  int getNSamples() const
  {
    return end - start;
  }

  int getDepth() const
  {
    return depth;
  }

  const SplitData &getSplitData() const
  {
    return splitData;
  }
  const Prediction &getPrediction() const
  {
    return prediction;
  }

  int getLeft(int dad) const
  {
    return dad*2 + 1;
  }
  
  int getRight(int dad) const
  {
    return getLeft(dad) + 1;
  }

  // ====================  MUTATORS      =======================================

  void setSplitData(SplitData splitData)
  {
    this->splitData = splitData;
  }
  void setPrediction(Prediction prediction)
  {
    this->prediction = prediction;
  }

  void setDepth(uint16_t depth)
  {
    this->depth = depth;
  }

  void setEnd(uint32_t end)
  {
    this->end = end;
  }

  void setStart(uint32_t start)
  {
    this->start = start;
  }

  void split(uint32_t start, uint32_t middle)
  {
  }

  // ====================  OPERATORS     =======================================

protected:
  // ====================  METHODS       =======================================

  // ====================  DATA MEMBERS  =======================================

private:
  // ====================  METHODS       =======================================

  // ====================  DATA MEMBERS  =======================================

public:
  SplitData splitData;
  Prediction prediction;
  uint32_t start, end;
  uint16_t depth;
  uint32_t idx;

};
// -----  end of class TNode  -----

template<class Sample, class Label>
struct LabelledSample
{
  Sample sample;
  Label label;
};

enum SplitResult
{
  SR_LEFT = 0, SR_RIGHT = 1, SR_INVALID = 2
};

typedef vector<SplitResult> SplitResultsVector;

// =====================================================================================
//        Class:  RandomTree
//  Description:
// =====================================================================================
template<class SplitData, class Sample, class Label, class Prediction, class ErrorData>
class RandomTree
{
public:
  typedef LabelledSample<Sample, Label> LSample;
  typedef vector<LSample> LSamplesVector;
  // ====================  LIFECYCLE     =======================================
  RandomTree()
  {
  }

  virtual ~RandomTree()
  {
    delete &heap;
  }

  // ====================  ACCESSORS     =======================================
  bool isLeaf( int nId ) const
  {
	  return ( heap[nId].getLeft(nId) >= heap.size() );
  }
  void save(string filename, bool includeSamples = false) const
  {
    ofstream out(filename.c_str());
    if (out.is_open()==false)
    {
        cout<<"Failed to open "<<filename<<endl;
        return;
    }
    writeHeader(out);
    out << endl;
    write(heap[0], out);
    out << includeSamples << " ";
    if (includeSamples)
       write(samples, out);
  }

  Prediction predict(Sample &sample) const
  {
  }


  // ====================  MUTATORS      =======================================
// TRAIN

  void load(string filename)
  {
    ifstream in(filename.c_str());
    readHeader(in);
    heap[0] = TNode<SplitData, Prediction>(0, 0);
    read(&heap[0], in);
    bool includeSamples;
    in >> includeSamples;
    if (includeSamples)
      read(this->samples, in);
  }

  // ====================  OPERATORS     =======================================

protected:
  // ====================  METHODS       =======================================

  //virtual SplitData generateSplit(const TNode<SplitData, Prediction> *node) const=0;

  virtual SplitResult split(const SplitData &splitData, Sample &sample) const =0;

  virtual bool split(const TNode<SplitData, Prediction> *node, SplitData &splitData,
      Prediction &leftPrediction, Prediction &rightPrediction) = 0;

  virtual void initialize(const TNode<SplitData, Prediction> *node, ErrorData &errorData,
      Prediction &prediction) const = 0;

  virtual void updateError(ErrorData &newError, const ErrorData &errorData,
      int nId, Prediction &newLeft,
      Prediction &newRight) const = 0;

  virtual double getError(const ErrorData &error) const = 0;

  // non-pure virtual function which allows to modify predictions after all node split trials are made
  virtual bool updateLeafPrediction(const TNode<SplitData, Prediction> *node, Prediction &newPrediction) const
  {
    return false;
  }

  const LSamplesVector &getLSamples() const
  {
    return samples;
  }

  LSamplesVector &getLSamples()
  {
    return samples;
  }

  SplitResultsVector &getSplitResults()
  {
    return splitResults;
  }

  const SplitResultsVector &getSplitResults() const
  {
    return splitResults;
  }

  TNode<SplitData,Prediction>* getRoot() const
  {
    return &heap[0];
  }

  virtual void writeHeader(ostream &out) const=0;
  virtual void readHeader(istream &in) =0;

  virtual void write(const Sample &sample, ostream &out) const =0;
  virtual void read(Sample &sample, istream &in) const =0;

  virtual void write(const Prediction &prediction, ostream &out) const=0;
  virtual void read(Prediction &prediction, istream &in) const=0;

  virtual void write(const Label &label, ostream &out) const=0;
  virtual void read(Label &label, istream &in) const=0;

  virtual void write(const SplitData &splitData, ostream &out) const=0;
  virtual void read(SplitData &splitData, istream &in) const=0;

  // ====================  DATA MEMBERS  =======================================

protected:
  // ====================  METHODS       =======================================

  //bool tryImprovingSplit(ErrorData &errorData, TNode<SplitData, Prediction> *node)
  //{
  //}

  void doSplit(const TNode<SplitData, Prediction> *node, int &pInvalid, int &pLeft)
  {
    pLeft = node->getStart();
    pInvalid = node->getStart();

    int pRight = node->getEnd() - 1;
    while (pLeft <= pRight)
    {
      LSample s;
      switch (splitResults[pLeft])
      {
      case SR_RIGHT:
        s = samples[pRight];
        samples[pRight] = samples[pLeft];
        samples[pLeft] = s;
        splitResults[pLeft] = splitResults[pRight];
        splitResults[pRight] = SR_RIGHT; //not necessary
        --pRight;
        break;
      case SR_INVALID:
        s = samples[pInvalid];
        samples[pInvalid] = samples[pLeft];
        samples[pLeft] = s;

        splitResults[pLeft] = splitResults[pInvalid];
        splitResults[pInvalid] = SR_INVALID;

        ++pInvalid;
        ++pLeft;
        break;
      case SR_LEFT:
        ++pLeft;
        break;
      }
    }
  }

  virtual void write(const TNode<SplitData, Prediction> *node, ostream &out) const
  {

  }

  virtual void read(TNode<SplitData, Prediction> *node, istream &in) const
  {

  }

  void write(const LSamplesVector &lSamples, ostream &out) const
  {
    out << lSamples.size() << " ";
    for (int i = 0; i < lSamples.size(); ++i)
    {
      write(lSamples[i].sample, out);
      out << " ";
      write(lSamples[i].label, out);
      out << " ";
    }
  }

  void read(LSamplesVector &lSamples, istream &in) const
  {
    int nSamples;
    in >> nSamples;
    lSamples.resize(nSamples);
    for (int i = 0; i < nSamples; ++i)
    {
      read(lSamples[i].sample, in);
      read(lSamples[i].label, in);
    }
  }

// ====================  DATA MEMBERS  =======================================

  vector<TNode<SplitData, Prediction>> heap;
  LSamplesVector samples;
  SplitResultsVector splitResults;

  SplitData cSplitData;
  Prediction cLeftPrediction, cRightPrediction;
};
// -----  end of class RandomTree  -----

}
#endif
