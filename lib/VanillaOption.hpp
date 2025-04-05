#ifndef _VANILLA_OPTION_H
#define _VANILLA_OPTION_H

class VanillaOption {
private:
  void init();
  void copy(const VanillaOption& rhs);

  double K;
  double r;
  double T;
  double S;
  double sigma;
public:
  VanillaOption(); // Default Constructor 
  VanillaOption(const double& K, const double& _r,
		const double& _T, const double& _s,
		const double& _sigma);
  VanillaOption(const VanillaOption& rhs);
  VanillaOption& operator=(const VanillaOption rhs); // assignment operator
  virtual ~VanillaOption(); // Destructor is virtual
    
};

#endif
